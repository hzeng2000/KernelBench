import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_conv3d_mul_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gM: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    depth: int, height: int, width: int, kernel_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_d = bidx * bdimx + tidx
    out_h = bidy * bdimy + tidy
    out_w = bidz * bdimz + tidz
    
    if out_d < depth and out_h < height and out_w < width:
        pad = kernel_size // 2
        for n in range(batch_size):
            for oc in range(out_channels):
                acc = 0.0
                for ic in range(in_channels):
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                in_d = out_d + kd - pad
                                in_h = out_h + kh - pad
                                in_w = out_w + kw - pad
                                
                                if in_d >= 0 and in_d < depth and in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                                    x_val = gX[n, ic, in_d, in_h, in_w]
                                    w_val = gW[oc, ic, kd, kh, kw]
                                    acc += x_val * w_val
                
                out_val = acc + gB[oc]
                out_val = out_val * gM[oc, 0, 0, 0]
                gOut[n, oc, out_d, out_h, out_w] = out_val

@cute.kernel
def fused_norm_clamp_mul_kernel(
    gX: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor, gM: cute.Tensor, gOut: cute.Tensor,
    clamp_min: float, clamp_max: float, batch_size: int, channels: int,
    depth: int, height: int, width: int, eps: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    d = bidx * bdimx + tidx
    h = bidy * bdimy + tidy
    w = bidz * bdimz + tidz
    
    if d < depth and h < height and w < width:
        for n in range(batch_size):
            for c in range(channels):
                x_val = gX[n, c, d, h, w]
                mean = gMean[n, c, 0, 0, 0]
                var = gVar[n, c, 0, 0, 0]
                
                norm_val = (x_val - mean) / cute.sqrt(var + eps)
                
                clamped = cute.fminf(cute.fmaxf(norm_val, clamp_min), clamp_max)
                
                out_val = clamped * gM[c, 0, 0, 0]
                gOut[n, c, d, h, w] = out_val

@cute.kernel
def reduce_max_kernel(
    gX: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    d = bidx * bdimx + tidx
    h = bidy * bdimy + tidy
    w = bidz * bdimz + tidz
    
    if d < depth and h < height and w < width:
        for n in range(batch_size):
            max_val = -1e38
            for c in range(channels):
                val = gX[n, c, d, h, w]
                if val > max_val:
                    max_val = val
            gOut[n, d, h, w] = max_val

@cute.jit
def fused_conv3d_mul_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mM: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    depth: int, height: int, width: int, kernel_size: int
):
    threads_per_block = 8
    grid_d = cute.ceil_div(depth, threads_per_block)
    grid_h = cute.ceil_div(height, threads_per_block)
    grid_w = cute.ceil_div(width, threads_per_block)
    
    fused_conv3d_mul_kernel(
        mX, mW, mB, mM, mOut,
        batch_size, in_channels, out_channels,
        depth, height, width, kernel_size
    ).launch(
        grid=(grid_d, grid_h, grid_w),
        block=(threads_per_block, threads_per_block, threads_per_block)
    )

@cute.jit
def fused_norm_clamp_mul_host(
    mX: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor, mM: cute.Tensor, mOut: cute.Tensor,
    clamp_min: float, clamp_max: float, batch_size: int, channels: int,
    depth: int, height: int, width: int, eps: float
):
    threads_per_block = 8
    grid_d = cute.ceil_div(depth, threads_per_block)
    grid_h = cute.ceil_div(height, threads_per_block)
    grid_w = cute.ceil_div(width, threads_per_block)
    
    fused_norm_clamp_mul_kernel(
        mX, mMean, mVar, mM, mOut,
        clamp_min, clamp_max, batch_size, channels,
        depth, height, width, eps
    ).launch(
        grid=(grid_d, grid_h, grid_w),
        block=(threads_per_block, threads_per_block, threads_per_block)
    )

@cute.jit
def reduce_max_host(
    mX: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int
):
    threads_per_block = 8
    grid_d = cute.ceil_div(depth, threads_per_block)
    grid_h = cute.ceil_div(height, threads_per_block)
    grid_w = cute.ceil_div(width, threads_per_block)
    
    reduce_max_kernel(
        mX, mOut,
        batch_size, channels, depth, height, width
    ).launch(
        grid=(grid_d, grid_h, grid_w),
        block=(threads_per_block, threads_per_block, threads_per_block)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.eps = 1e-5
        self.compiled = {}

    def forward(self, x):
        batch_size, in_channels, depth, height, width = x.shape
        out_channels = self.conv_weight.shape[0]
        
        x = x.contiguous().cuda()
        conv_out = torch.empty(batch_size, out_channels, depth, height, width, dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(self.conv_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mB = from_dlpack(self.conv_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mM = from_dlpack(self.multiplier, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_conv3d_mul_host, mX, mW, mB, mM, mConvOut,
                                  batch_size, in_channels, out_channels, depth, height, width, 3)
            self.compiled[key] = compiled
        
        compiled(mX, mW, mB, mM, mConvOut, batch_size, in_channels, out_channels, depth, height, width, 3)
        
        mean = conv_out.mean(dim=(2, 3, 4), keepdim=True)
        var = conv_out.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        
        norm_out = torch.empty_like(conv_out)
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mNormOut = from_dlpack(norm_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        compiled_norm = cute.compile(fused_norm_clamp_mul_host, mConvOut, mMean, mVar, mM, mNormOut,
                                   self.clamp_min, self.clamp_max, batch_size, out_channels, depth, height, width, self.eps)
        compiled_norm(mConvOut, mMean, mVar, mM, mNormOut, self.clamp_min, self.clamp_max, batch_size, out_channels, depth, height, width, self.eps)
        
        max_out = torch.empty(batch_size, depth, height, width, dtype=x.dtype, device=x.device)
        mMaxOut = from_dlpack(max_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        compiled_max = cute.compile(reduce_max_host, mNormOut, mMaxOut, batch_size, out_channels, depth, height, width)
        compiled_max(mNormOut, mMaxOut, batch_size, out_channels, depth, height, width)
        
        return max_out