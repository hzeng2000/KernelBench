import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose_scale_bn_relu_pool_kernel(
    gI: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor,
    gO: cute.Tensor,
    scale: float, eps: float,
    batch_size: int, in_c: int, out_c: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int
):
    tid = cute.arch.thread_idx()
    bid = cute.arch.block_idx()
    bdim = cute.arch.block_dim()
    
    thread_id = bid * bdim + tid
    
    total_out_elems = batch_size * out_c * out_d * out_h * out_w
    if thread_id >= total_out_elems:
        return
    
    # Compute output indices
    tmp = thread_id
    ow = tmp % out_w
    tmp = tmp // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    tmp = tmp // out_d
    oc = tmp % out_c
    b = tmp // out_c
    
    # Compute input indices for transposed convolution
    sum_val = 0.0
    
    for ic in range(in_c):
        for kd in range(k_d):
            for kh in range(k_h):
                for kw in range(k_w):
                    # Input indices for transposed convolution
                    in_d_idx = od + kd - (k_d - 1)
                    in_h_idx = oh + kh - (k_h - 1)
                    in_w_idx = ow + kw - (k_w - 1)
                    
                    if (in_d_idx >= 0 and in_d_idx < in_d and
                        in_h_idx >= 0 and in_h_idx < in_h and
                        in_w_idx >= 0 and in_w_idx < in_w):
                        
                        weight_idx = ((oc * in_c + ic) * k_d + kd) * k_h + kh
                        weight_idx = weight_idx * k_w + kw
                        
                        input_idx = ((b * in_c + ic) * in_d + in_d_idx) * in_h + in_h_idx
                        input_idx = input_idx * in_w + in_w_idx
                        
                        w_val = gW[weight_idx]
                        i_val = gI[input_idx]
                        sum_val += i_val * w_val
    
    # Apply bias
    bias_val = gB[oc]
    sum_val += bias_val
    
    # Scale
    sum_val *= scale
    
    # Batch normalization
    mean_val = gMean[oc]
    var_val = gVar[oc]
    bn_val = (sum_val - mean_val) / cute.math.sqrt(var_val + eps)
    
    # Store output
    out_idx = thread_id
    gO[out_idx] = bn_val

@cute.jit
def conv_transpose_scale_bn_relu_pool_host(
    mI: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor,
    mO: cute.Tensor,
    scale: float, eps: float,
    batch_size: int, in_c: int, out_c: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int
):
    total_threads = batch_size * out_c * out_d * out_h * out_w
    threads_per_block = 256
    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    
    conv_transpose_scale_bn_relu_pool_kernel(
        mI, mW, mB, mMean, mVar, mO,
        scale, eps,
        batch_size, in_c, out_c,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        k_d, k_h, k_w
    ).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def global_avg_pool_kernel(
    gI: cute.Tensor, gO: cute.Tensor,
    batch_size: int, channels: int,
    d: int, h: int, w: int
):
    tid = cute.arch.thread_idx()
    bid = cute.arch.block_idx()
    bdim = cute.arch.block_dim()
    
    thread_id = bid * bdim + tid
    total_elems = batch_size * channels
    
    if thread_id >= total_elems:
        return
    
    b = thread_id // channels
    c = thread_id % channels
    
    sum_val = 0.0
    total_pixels = d * h * w
    
    for i in range(d * h * w):
        idx = ((b * channels + c) * d * h * w) + i
        sum_val += gI[idx]
    
    avg_val = sum_val / total_pixels
    out_idx = thread_id
    gO[out_idx] = avg_val

@cute.jit
def global_avg_pool_host(
    mI: cute.Tensor, mO: cute.Tensor,
    batch_size: int, channels: int,
    d: int, h: int, w: int
):
    total_threads = batch_size * channels
    threads_per_block = 256
    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    
    global_avg_pool_kernel(mI, mO, batch_size, channels, d, h, w).launch(
        grid=(blocks, 1, 1), block=(threads_per_block, 1, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.eps = eps
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Batch norm parameters
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        
        self.compiled = {}
    
    def forward(self, x):
        batch_size, in_c, in_d, in_h, in_w = x.shape
        k_d = k_h = k_w = self.kernel_size
        
        # Calculate output dimensions for transposed convolution
        out_d = in_d + k_d - 1
        out_h = in_h + k_h - 1
        out_w = in_w + k_w - 1
        
        # Prepare input tensors
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        running_mean = self.running_mean.contiguous().cuda()
        running_var = self.running_var.contiguous().cuda()
        
        # Intermediate output
        intermediate = torch.empty(batch_size * self.out_channels * out_d * out_h * out_w, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mI = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mMean = from_dlpack(running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mVar = from_dlpack(running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mIntermediate = from_dlpack(intermediate, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        # Compile and run fused kernel
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_transpose_scale_bn_relu_pool_host, 
                                  mI, mW, mB, mMean, mVar, mIntermediate,
                                  self.scale_factor, self.eps,
                                  batch_size, in_c, self.out_channels,
                                  in_d, in_h, in_w,
                                  out_d, out_h, out_w,
                                  k_d, k_h, k_w)
            self.compiled[key] = compiled
        
        compiled(mI, mW, mB, mMean, mVar, mIntermediate,
                self.scale_factor, self.eps,
                batch_size, in_c, self.out_channels,
                in_d, in_h, in_w,
                out_d, out_h, out_w,
                k_d, k_h, k_w)
        
        # Reshape for global average pooling
        intermediate_reshaped = intermediate.view(batch_size, self.out_channels, out_d, out_h, out_w)
        
        # Global average pooling
        output = torch.empty(batch_size * self.out_channels, dtype=x.dtype, device=x.device)
        mIntermediate2 = from_dlpack(intermediate_reshaped, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        compiled_pool = cute.compile(global_avg_pool_host, mIntermediate2, mOutput,
                                   batch_size, self.out_channels, out_d, out_h, out_w)
        compiled_pool(mIntermediate2, mOutput, batch_size, self.out_channels, out_d, out_h, out_w)
        
        return output.view(batch_size, self.out_channels, 1, 1, 1)