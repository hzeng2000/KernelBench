import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose3d_gelu_ln_scale_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor,
    gOutput: cute.Tensor, gMean: cute.Tensor, gRstd: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    eps: float, scale: float
):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, tidy, tidz = cute.arch.thread_idx()
    
    # Grid-strided loop over output spatial locations
    out_idx = bidx * cute.arch.block_dim().x + tidx
    total_out_spatial = batch_size * out_d * out_h * out_w
    
    if out_idx < total_out_spatial:
        # Decompose flat index
        n = out_idx // (out_d * out_h * out_w)
        rem = out_idx % (out_d * out_h * out_w)
        od = rem // (out_h * out_w)
        rem = rem % (out_h * out_w)
        oh = rem // out_w
        ow = rem % out_w
        
        # Compute input region
        id_start = od * stride_d - pad_d
        ih_start = oh * stride_h - pad_h
        iw_start = ow * stride_w - pad_w
        
        # Accumulate over input channels and kernel
        sum_val = 0.0
        for c in range(in_channels):
            for kd in range(kernel_d):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        id_val = id_start + kd
                        ih_val = ih_start + kh
                        iw_val = iw_start + kw
                        
                        if id_val >= 0 and id_val < in_d and ih_val >= 0 and ih_val < in_h and iw_val >= 0 and iw_val < in_w:
                            inp_val = gInput[n, c, id_val, ih_val, iw_val]
                            w_val = gWeight[c, 0, kd, kh, kw]  # Weight layout: [in_c, out_c, kd, kh, kw] but out_c=0 for kernel
                            sum_val += inp_val * w_val
        
        # Add bias if present
        if gBias.shape[0] > 0:
            sum_val += gBias[0]
        
        # Store pre-norm value temporarily
        gOutput[n, 0, od, oh, ow] = sum_val

@cute.kernel
def layer_norm_gelu_scale_kernel(
    gOutput: cute.Tensor, gMean: cute.Tensor, gRstd: cute.Tensor,
    batch_size: int, out_channels: int, out_d: int, out_h: int, out_w: int,
    eps: float, scale: float
):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, tidy, tidz = cute.arch.thread_idx()
    
    # One thread per (n, spatial) location
    n = bidx
    spatial_idx = bidy * cute.arch.block_dim().x + tidx
    total_spatial = out_d * out_h * out_w
    
    if n < batch_size and spatial_idx < total_spatial:
        # Decompose spatial index
        od = spatial_idx // (out_h * out_w)
        rem = spatial_idx % (out_h * out_w)
        oh = rem // out_w
        ow = rem % out_w
        
        # Compute mean
        sum_val = 0.0
        for c in range(out_channels):
            sum_val += gOutput[n, c, od, oh, ow]
        mean = sum_val / out_channels
        gMean[n, od, oh, ow] = mean
        
        # Compute variance
        var_sum = 0.0
        for c in range(out_channels):
            diff = gOutput[n, c, od, oh, ow] - mean
            var_sum += diff * diff
        var = var_sum / out_channels
        rstd = cute.math.rsqrt(var + eps)
        gRstd[n, od, oh, ow] = rstd
        
        # Apply layer norm, GELU, and scale
        for c in range(out_channels):
            val = gOutput[n, c, od, oh, ow]
            normalized = (val - mean) * rstd
            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            x_cubed = normalized * normalized * normalized
            tanh_arg = 0.7978845608 * (normalized + 0.044715 * x_cubed)
            gelu_val = 0.5 * normalized * (1.0 + cute.math.tanh(tanh_arg))
            gOutput[n, c, od, oh, ow] = gelu_val * scale

@cute.jit
def fused_conv_transpose3d_ln_gelu_scale_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor,
    mOutput: cute.Tensor, mMean: cute.Tensor, mRstd: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_size: int, stride: int, padding: int,
    eps: float, scale: float
):
    kernel_d = kernel_h = kernel_w = kernel_size
    stride_d = stride_h = stride_w = stride
    pad_d = pad_h = pad_w = padding
    
    # Launch conv transpose kernel
    total_out_spatial = batch_size * out_d * out_h * out_w
    threads_per_block = 256
    grid_x = (total_out_spatial + threads_per_block - 1) // threads_per_block
    conv_transpose3d_gelu_ln_scale_kernel(
        mInput, mWeight, mBias, mOutput, mMean,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w, eps, scale
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))
    
    # Launch layer norm + GELU + scale kernel
    spatial_size = out_d * out_h * out_w
    grid_x_ln = batch_size
    grid_y_ln = (spatial_size + threads_per_block - 1) // threads_per_block
    layer_norm_gelu_scale_kernel(
        mOutput, mMean, mRstd,
        batch_size, out_channels, out_d, out_h, out_w,
        eps, scale
    ).launch(grid=(grid_x_ln, grid_y_ln, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.scaling_factor = scaling_factor
        
        # Initialize weight tensor [in_channels, 1, kernel, kernel, kernel] for kernel fusion
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)
        
        self.compiled = {}

    def forward(self, x):
        batch_size, _, in_d, in_h, in_w = x.shape
        
        # Compute output spatial dimensions
        out_d = (in_d - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        # Allocate output and intermediate tensors
        output = torch.empty(batch_size, self.out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        mean = torch.empty(batch_size, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        rstd = torch.empty(batch_size, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mInput = from_dlpack(x.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(self.weight.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(self.bias.contiguous() if self.bias is not None else torch.zeros(1, device=x.device), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mRstd = from_dlpack(rstd, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        # Compile and launch kernel
        key = (x.dtype, self.kernel_size, self.stride, self.padding)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_conv_transpose3d_ln_gelu_scale_host,
                mInput, mWeight, mBias, mOutput, mMean, mRstd,
                batch_size, self.in_channels, self.out_channels,
                in_d, in_h, in_w, out_d, out_h, out_w,
                self.kernel_size, self.stride, self.padding,
                self.eps, self.scaling_factor
            )
            self.compiled[key] = compiled
        
        compiled(mInput, mWeight, mBias, mOutput, mMean, mRstd,
                batch_size, self.in_channels, self.out_channels,
                in_d, in_h, in_w, out_d, out_h, out_w,
                self.kernel_size, self.stride, self.padding,
                self.eps, self.scaling_factor)
        
        return output