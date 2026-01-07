import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_transpose_conv_maxpool_softmax_sub_swish_max_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gSubtract: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int,
    pool_kd: int, pool_kh: int, pool_kw: int,
    pool_sd: int, pool_sh: int, pool_sw: int,
    pool_pd: int, pool_ph: int, pool_pw: int
):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, tidy, tidz = cute.arch.thread_idx()
    
    # Each thread processes one output spatial location
    out_x = bidx * cute.arch.block_dim_x() + tidx
    out_y = bidy * cute.arch.block_dim_y() + tidy
    out_z = bidz * cute.arch.block_dim_z() + tidz
    
    if out_x < out_h and out_y < out_w and out_z < out_d:
        # MaxPool output dimensions
        pool_out_d = (out_d + 2 * pool_pd - pool_kd) // pool_sd + 1
        pool_out_h = (out_h + 2 * pool_ph - pool_kh) // pool_sh + 1
        pool_out_w = (out_w + 2 * pool_pw - pool_kw) // pool_sw + 1
        
        # Shared memory for intermediate results
        smem = cute.shared_memory(size=16 * 32 * 32 * 4)  # out_channels * spatial_size
        
        # Compute ConvTranspose3d
        for b in range(batch_size):
            max_val = float('-inf')
            
            # Process each output channel
            for oc in range(out_channels):
                accum = 0.0
                
                # ConvTranspose computation
                for ic in range(in_channels):
                    for kd in range(kernel_d):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                # Input coordinates
                                in_z = (out_z + pad_d - kd * (kernel_d - 1) - out_pad_d) // stride_d
                                in_y = (out_y + pad_h - kh * (kernel_h - 1) - out_pad_h) // stride_h
                                in_x = (out_x + pad_w - kw * (kernel_w - 1) - out_pad_w) // stride_w
                                
                                if (in_z >= 0 and in_z < in_d and 
                                    in_y >= 0 and in_y < in_h and 
                                    in_x >= 0 and in_x < in_w and
                                    (out_z + pad_d - kd * (kernel_d - 1) - out_pad_d) % stride_d == 0 and
                                    (out_y + pad_h - kh * (kernel_h - 1) - out_pad_h) % stride_h == 0 and
                                    (out_x + pad_w - kw * (kernel_w - 1) - out_pad_w) % stride_w == 0):
                                    
                                    weight_idx = oc * in_channels * kernel_d * kernel_h * kernel_w + \
                                                ic * kernel_d * kernel_h * kernel_w + \
                                                kd * kernel_h * kernel_w + kh * kernel_w + kw
                                    input_idx = b * in_channels * in_d * in_h * in_w + \
                                               ic * in_d * in_h * in_w + \
                                               in_z * in_h * in_w + in_y * in_w + in_x
                                    accum += gInput[input_idx] * gWeight[weight_idx]
                
                # Add bias
                if gBias.shape[0] > 0:
                    accum += gBias[oc]
                
                # Store in shared memory
                smem[oc * 32 * 32 + out_z * 32 + out_y] = accum
            
            # MaxPool3d within the thread block
            pool_z = out_z // pool_sd
            pool_y = out_y // pool_sh
            pool_x = out_x // pool_sw
            
            if pool_z < pool_out_d and pool_y < pool_out_h and pool_x < pool_out_w:
                # Softmax across channels (dim=1)
                max_channel = float('-inf')
                for oc in range(out_channels):
                    val = smem[oc * 32 * 32 + out_z * 32 + out_y]
                    if val > max_channel:
                        max_channel = val
                
                exp_sum = 0.0
                for oc in range(out_channels):
                    val = smem[oc * 32 * 32 + out_z * 32 + out_y]
                    exp_val = math.exp(val - max_channel)
                    smem[oc * 32 * 32 + out_z * 32 + out_y] = exp_val
                    exp_sum += exp_val
                
                # Normalize
                for oc in range(out_channels):
                    smem[oc * 32 * 32 + out_z * 32 + out_y] /= exp_sum
                
                # Subtract across channels
                for oc in range(out_channels):
                    smem[oc * 32 * 32 + out_z * 32 + out_y] -= gSubtract[oc]
                
                # Swish activation
                for oc in range(out_channels):
                    val = smem[oc * 32 * 32 + out_z * 32 + out_y]
                    sigmoid = 1.0 / (1.0 + math.exp(-val))
                    smem[oc * 32 * 32 + out_z * 32 + out_y] = sigmoid * val
                
                # Max across channels
                channel_max = float('-inf')
                for oc in range(out_channels):
                    val = smem[oc * 32 * 32 + out_z * 32 + out_y]
                    if val > channel_max:
                        channel_max = val
                
                # Store final result
                out_idx = b * pool_out_d * pool_out_h * pool_out_w + \
                         pool_z * pool_out_h * pool_out_w + pool_y * pool_out_w + pool_x
                gOutput[out_idx] = channel_max

@cute.jit
def fused_transpose_conv_maxpool_softmax_sub_swish_max_host(
    input_tensor: cute.Tensor, weight_tensor: cute.Tensor, bias_tensor: cute.Tensor, subtract_tensor: cute.Tensor, output_tensor: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int,
    pool_kd: int, pool_kh: int, pool_kw: int,
    pool_sd: int, pool_sh: int, pool_sw: int,
    pool_pd: int, pool_ph: int, pool_pw: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(out_h, 8)
    grid_y = cute.ceil_div(out_w, 8)
    grid_z = cute.ceil_div(out_d, 8)
    
    fused_transpose_conv_maxpool_softmax_sub_swish_max_kernel(
        input_tensor, weight_tensor, bias_tensor, subtract_tensor, output_tensor,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w,
        pool_kd, pool_kh, pool_kw, pool_sd, pool_sh, pool_sw,
        pool_pd, pool_ph, pool_pw
    ).launch(grid=(grid_x, grid_y, grid_z), block=(8, 8, 8))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.pool_kernel_size = pool_kernel_size if isinstance(pool_kernel_size, tuple) else (pool_kernel_size, pool_kernel_size, pool_kernel_size)
        self.pool_stride = pool_stride if isinstance(pool_stride, tuple) else (pool_stride, pool_stride, pool_stride)
        self.pool_padding = pool_padding if isinstance(pool_padding, tuple) else (pool_padding, pool_padding, pool_padding)
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.subtract = nn.Parameter(torch.randn(out_channels))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.compiled = None

    def forward(self, x):
        batch_size = x.shape[0]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        
        # Calculate output dimensions
        out_d = (in_d - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        out_h = (in_h - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        out_w = (in_w - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2] + self.output_padding[2]
        
        # Calculate pooled dimensions
        pool_out_d = (out_d + 2 * self.pool_padding[0] - self.pool_kernel_size[0]) // self.pool_stride[0] + 1
        pool_out_h = (out_h + 2 * self.pool_padding[1] - self.pool_kernel_size[1]) // self.pool_stride[1] + 1
        pool_out_w = (out_w + 2 * self.pool_padding[2] - self.pool_kernel_size[2]) // self.pool_stride[2] + 1
        
        # Prepare tensors
        x_contig = x.contiguous().cuda()
        weight_contig = self.weight.contiguous().cuda()
        bias_contig = self.bias.contiguous().cuda()
        subtract_contig = self.subtract.contiguous().cuda()
        
        output = torch.empty(batch_size, pool_out_d, pool_out_h, pool_out_w, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mInput = from_dlpack(x_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mSubtract = from_dlpack(subtract_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        # Compile if not already done
        if self.compiled is None:
            self.compiled = cute.compile(
                fused_transpose_conv_maxpool_softmax_sub_swish_max_host,
                mInput, mWeight, mBias, mSubtract, mOutput,
                batch_size, self.in_channels, self.out_channels,
                in_d, in_h, in_w, out_d, out_h, out_w,
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.stride[0], self.stride[1], self.stride[2],
                self.padding[0], self.padding[1], self.padding[2],
                self.output_padding[0], self.output_padding[1], self.output_padding[2],
                self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2],
                self.pool_stride[0], self.pool_stride[1], self.pool_stride[2],
                self.pool_padding[0], self.pool_padding[1], self.pool_padding[2]
            )
        
        # Launch kernel
        self.compiled(
            mInput, mWeight, mBias, mSubtract, mOutput,
            batch_size, self.in_channels, self.out_channels,
            in_d, in_h, in_w, out_d, out_h, out_w,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2],
            self.pool_stride[0], self.pool_stride[1], self.pool_stride[2],
            self.pool_padding[0], self.pool_padding[1], self.pool_padding[2]
        )
        
        return output