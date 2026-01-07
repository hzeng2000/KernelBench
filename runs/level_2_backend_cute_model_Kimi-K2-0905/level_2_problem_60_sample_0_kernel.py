import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_conv_transpose_swish_gn_hardswish_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    groups: int, eps: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    thread_idx = (bidz * bdimz + tidz) * bdimx * bdimy + (bidy * bdimy + tidy) * bdimx + (bidx * bdimx + tidx)
    
    total_threads = batch_size * out_channels * out_d * out_h * out_w
    if thread_idx >= total_threads:
        return
    
    # Compute output position
    tmp = thread_idx
    n = tmp // (out_channels * out_d * out_h * out_w)
    tmp %= (out_channels * out_d * out_h * out_w)
    c_out = tmp // (out_d * out_h * out_w)
    tmp %= (out_d * out_h * out_w)
    d_out = tmp // (out_h * out_w)
    tmp %= (out_h * out_w)
    h_out = tmp // out_w
    w_out = tmp % out_w
    
    # Compute group norm parameters
    channels_per_group = out_channels // groups
    group_idx = c_out // channels_per_group
    
    # Compute convolution transpose
    acc = 0.0
    
    for c_in in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    d_in = (d_out + pad_d - kd) // stride_d
                    h_in = (h_out + pad_h - kh) // stride_h
                    w_in = (w_out + pad_w - kw) // stride_w
                    
                    if (d_out + pad_d - kd) % stride_d == 0 and \
                       (h_out + pad_h - kh) % stride_h == 0 and \
                       (w_out + pad_w - kw) % stride_w == 0 and \
                       d_in >= 0 and d_in < in_d and \
                       h_in >= 0 and h_in < in_h and \
                       w_in >= 0 and w_in < in_w:
                        
                        weight_val = gWeight[c_out, c_in, kd, kh, kw]
                        input_val = gInput[n, c_in, d_in, h_in, w_in]
                        acc += weight_val * input_val
    
    if gBias.shape[0] > 0:
        acc += gBias[c_out]
    
    # Swish activation
    swish_val = acc * (1.0 / (1.0 + cute.math.exp(-acc)))
    
    # Store intermediate for group norm
    gOutput[n, c_out, d_out, h_out, w_out] = swish_val

@cute.kernel
def group_norm_hardswish_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, channels: int, groups: int, eps: float,
    d: int, h: int, w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    thread_idx = (bidz * bdimz + tidz) * bdimx * bdimy + (bidy * bdimy + tidy) * bdimx + (bidx * bdimx + tidx)
    
    total_threads = batch_size * groups * d * h * w
    if thread_idx >= total_threads:
        return
    
    # Compute position
    tmp = thread_idx
    n = tmp // (groups * d * h * w)
    tmp %= (groups * d * h * w)
    g = tmp // (d * h * w)
    tmp %= (d * h * w)
    d_idx = tmp // (h * w)
    tmp %= (h * w)
    h_idx = tmp // w
    w_idx = tmp % w
    
    channels_per_group = channels // groups
    start_c = g * channels_per_group
    end_c = start_c + channels_per_group
    
    # Compute mean
    sum_val = 0.0
    count = channels_per_group * d * h * w
    
    for c in range(start_c, end_c):
        for dd in range(d):
            for hh in range(h):
                for ww in range(w):
                    sum_val += gInput[n, c, dd, hh, ww]
    
    mean = sum_val / count
    
    # Compute variance
    var_sum = 0.0
    for c in range(start_c, end_c):
        for dd in range(d):
            for hh in range(h):
                for ww in range(w):
                    diff = gInput[n, c, dd, hh, ww] - mean
                    var_sum += diff * diff
    
    var = var_sum / count
    std = cute.math.sqrt(var + eps)
    
    # Normalize and apply HardSwish
    for c in range(start_c, end_c):
        val = gInput[n, c, d_idx, h_idx, w_idx]
        normalized = (val - mean) / std
        # HardSwish: x * relu6(x + 3) / 6
        hardswish_val = normalized * cute.math.min(cute.math.max(normalized + 3.0, 0.0), 6.0) / 6.0
        gOutput[n, c, d_idx, h_idx, w_idx] = hardswish_val

@cute.jit
def fused_conv_transpose_swish_gn_hardswish_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mIntermediate: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    groups: int, eps: float
):
    total_threads = batch_size * out_channels * out_d * out_h * out_w
    threads_per_block = 256
    grid_size = cute.ceil_div(total_threads, threads_per_block)
    
    fused_conv_transpose_swish_gn_hardswish_kernel(
        mInput, mWeight, mBias, mIntermediate,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups, eps
    ).launch(grid=(grid_size, 1, 1), block=(threads_per_block, 1, 1))

@cute.jit
def group_norm_hardswish_host(
    mInput: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, channels: int, groups: int, eps: float,
    d: int, h: int, w: int
):
    total_threads = batch_size * groups * d * h * w
    threads_per_block = 256
    grid_size = cute.ceil_div(total_threads, threads_per_block)
    
    group_norm_hardswish_kernel(
        mInput, mOutput,
        batch_size, channels, groups, eps,
        d, h, w
    ).launch(grid=(grid_size, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.groups = groups
        self.eps = eps
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        
        self.reset_parameters()
        self.compiled = {}
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, in_d, in_h, in_w = x.shape
        out_d = (in_d - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        out_h = (in_h - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
        out_w = (in_w - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2]
        
        x = x.contiguous().cuda()
        intermediate = torch.empty((batch_size, self.out_channels, out_d, out_h, out_w), dtype=x.dtype, device=x.device)
        output = torch.empty((batch_size, self.out_channels, out_d, out_h, out_w), dtype=x.dtype, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(self.bias if self.bias is not None else torch.empty(0, device=x.device), assumed_align=16)
        mIntermediate = from_dlpack(intermediate, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype, batch_size, in_channels, self.out_channels, out_d, out_h, out_w)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_conv_transpose_swish_gn_hardswish_host, 
                                    mInput, mWeight, mBias, mIntermediate, mOutput,
                                    batch_size, in_channels, self.out_channels,
                                    in_d, in_h, in_w,
                                    out_d, out_h, out_w,
                                    self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                                    self.stride[0], self.stride[1], self.stride[2],
                                    self.padding[0], self.padding[1], self.padding[2],
                                    self.groups, self.eps)
            self.compiled[key] = compiled
        
        compiled(mInput, mWeight, mBias, mIntermediate, mOutput,
                 batch_size, in_channels, self.out_channels,
                 in_d, in_h, in_w,
                 out_d, out_h, out_w,
                 self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                 self.stride[0], self.stride[1], self.stride[2],
                 self.padding[0], self.padding[1], self.padding[2],
                 self.groups, self.eps)
        
        key2 = (x.dtype, batch_size, self.out_channels, self.groups, out_d, out_h, out_w)
        compiled2 = self.compiled.get(key2)
        if compiled2 is None:
            compiled2 = cute.compile(group_norm_hardswish_host,
                                     mIntermediate, mOutput,
                                     batch_size, self.out_channels, self.groups, self.eps,
                                     out_d, out_h, out_w)
            self.compiled[key2] = compiled2
        
        compiled2(mIntermediate, mOutput,
                  batch_size, self.out_channels, self.groups, self.eps,
                  out_d, out_h, out_w)
        
        return output