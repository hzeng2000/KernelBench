import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose_bias_gmp_lse_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, in_h: int, in_w: int, out_h: int, out_w: int,
    in_c: int, out_c: int, k: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_n = bidz * bdimz + tidz
    
    if out_n < batch_size and out_y < out_h and out_x < out_w:
        sum_val = 0.0
        for ic in range(in_c):
            for kh in range(k):
                for kw in range(k):
                    in_y = out_y - kh
                    in_x = out_x - kw
                    if in_y >= 0 and in_y < in_h and in_x >= 0 and in_x < in_w:
                        w_idx = ic * out_c * k * k + (out_c - 1 - (out_n * out_c * out_h * out_w + out_y * out_w + out_x) % out_c) * k * k + kh * k + kw
                        x_idx = out_n * in_c * in_h * in_w + ic * in_h * in_w + in_y * in_w + in_x
                        sum_val += gX[x_idx] * gW[w_idx]
        
        out_idx = out_n * out_c * out_h * out_w + ((out_n * out_c * out_h * out_w + out_y * out_w + out_x) % out_c) * out_h * out_w + out_y * out_w + out_x
        gOut[out_idx] = sum_val + gB[(out_n * out_c * out_h * out_w + out_y * out_w + out_x) % out_c]

@cute.kernel
def global_mean_pool_kernel(
    gIn: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, channels: int, h: int, w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    n = bidz * bdimz + tidz
    c = bidx * bdimx + tidx
    
    if n < batch_size and c < channels:
        sum_val = 0.0
        for y in range(h):
            for x in range(w):
                idx = n * channels * h * w + c * h * w + y * w + x
                sum_val += gIn[idx]
        mean_val = sum_val / (h * w)
        out_idx = n * channels + c
        gOut[out_idx] = mean_val

@cute.kernel
def logsumexp_kernel(
    gIn: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, channels: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    n = bidz * bdimz + tidz
    
    if n < batch_size:
        max_val = -1e38
        for c in range(channels):
            idx = n * channels + c
            val = gIn[idx]
            if val > max_val:
                max_val = val
        
        sum_exp = 0.0
        for c in range(channels):
            idx = n * channels + c
            sum_exp += cute.exp(gIn[idx] - max_val)
        
        gOut[n] = max_val + cute.log(sum_exp)

@cute.jit
def fused_conv_transpose_bias_gmp_lse_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, in_h: int, in_w: int, out_h: int, out_w: int,
    in_c: int, out_c: int, k: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(out_h * out_w, threads_per_block)
    grid_y = 1
    grid_z = batch_size
    
    conv_transpose_bias_gmp_lse_kernel(
        mX, mW, mB, mOut,
        batch_size, in_h, in_w, out_h, out_w,
        in_c, out_c, k
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

@cute.jit
def global_mean_pool_host(
    mIn: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, channels: int, h: int, w: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(channels, threads_per_block)
    grid_y = 1
    grid_z = batch_size
    
    global_mean_pool_kernel(
        mIn, mOut,
        batch_size, channels, h, w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

@cute.jit
def logsumexp_host(
    mIn: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, channels: int
):
    threads_per_block = 1
    grid_x = 1
    grid_y = 1
    grid_z = batch_size
    
    logsumexp_kernel(
        mIn, mOut,
        batch_size, channels
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_h, in_w = x.shape[2], x.shape[3]
        out_h = in_h + self.kernel_size - 1
        out_w = in_w + self.kernel_size - 1
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        conv_out = torch.empty((batch_size, self.out_channels, out_h, out_w), dtype=x.dtype, device=x.device)
        pooled_out = torch.empty((batch_size, self.out_channels), dtype=x.dtype, device=x.device)
        lse_out = torch.empty((batch_size,), dtype=x.dtype, device=x.device)
        final_out = torch.empty((batch_size,), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mPooledOut = from_dlpack(pooled_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mLseOut = from_dlpack(lse_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mFinalOut = from_dlpack(final_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled_conv = cute.compile(fused_conv_transpose_bias_gmp_lse_host, mX, mW, mB, mConvOut,
                                       batch_size, in_h, in_w, out_h, out_w,
                                       self.in_channels, self.out_channels, self.kernel_size)
            compiled_pool = cute.compile(global_mean_pool_host, mConvOut, mPooledOut,
                                       batch_size, self.out_channels, out_h, out_w)
            compiled_lse = cute.compile(logsumexp_host, mPooledOut, mLseOut,
                                      batch_size, self.out_channels)
            self.compiled[key] = (compiled_conv, compiled_pool, compiled_lse)
        else:
            compiled_conv, compiled_pool, compiled_lse = compiled
        
        compiled_conv(mX, mW, mB, mConvOut,
                     batch_size, in_h, in_w, out_h, out_w,
                     self.in_channels, self.out_channels, self.kernel_size)
        
        compiled_pool(mConvOut, mPooledOut,
                      batch_size, self.out_channels, out_h, out_w)
        
        pooled_plus_bias = pooled_out + bias.squeeze()
        
        mPooledBias = from_dlpack(pooled_plus_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        compiled_lse(mPooledBias, mLseOut,
                     batch_size, self.out_channels)
        
        final_out = lse_out * 10.0
        
        return final_out