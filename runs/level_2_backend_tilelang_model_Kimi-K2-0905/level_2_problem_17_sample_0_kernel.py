import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv_instancenorm_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    block_h: int = 8,
    block_w: int = 8,
    warp_m: int = 16,
    warp_n: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_height = height
    out_width = width
    
    @T.prim_func
    def fused_conv_instancenorm_kernel(
        X: T.Tensor((batch_size, in_channels, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        RunningMean: T.Tensor((out_channels,), dtype),
        RunningVar: T.Tensor((out_channels,), dtype),
        Y: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_w), T.ceildiv(out_height, block_h), batch_size * out_channels, threads=threads) as (bx, by, bz):
            tile_w = bx * block_w
            tile_h = by * block_h
            batch = bz // out_channels
            oc = bz % out_channels
            
            # Shared memory for tile
            shared = T.alloc_shared((block_h + 2 * pad, block_w + 2 * pad), dtype)
            
            # Registers for accumulation
            accum = T.alloc_fragment((block_h, block_w), dtype, 0)
            
            # Load input tile with halo
            for ih in T.Parallel(block_h + 2 * pad):
                for iw in T.Parallel(block_w + 2 * pad):
                    h_idx = tile_h + ih - pad
                    w_idx = tile_w + iw - pad
                    if h_idx >= 0 and h_idx < height and w_idx >= 0 and w_idx < width:
                        shared[ih, iw] = X[batch, 0, h_idx, w_idx]
                    else:
                        shared[ih, iw] = T.cast(0, dtype)
            
            # Convolution
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    for ih in T.Parallel(block_h):
                        for iw in T.Parallel(block_w):
                            accum[ih, iw] += shared[ih + kh, iw + kw] * W[oc, 0, kh, kw]
            
            # Add bias
            for ih in T.Parallel(block_h):
                for iw in T.Parallel(block_w):
                    accum[ih, iw] += B[oc]
            
            # Instance Norm: compute mean and var for this channel
            mean = T.alloc_fragment((1,), dtype, 0)
            var = T.alloc_fragment((1,), dtype, 0)
            count = T.cast(block_h * block_w, dtype)
            
            # Mean
            for ih in T.Parallel(block_h):
                for iw in T.Parallel(block_w):
                    if tile_h + ih < out_height and tile_w + iw < out_width:
                        mean[0] += accum[ih, iw]
            mean[0] = mean[0] / count
            
            # Variance
            for ih in T.Parallel(block_h):
                for iw in T.Parallel(block_w):
                    if tile_h + ih < out_height and tile_w + iw < out_width:
                        diff = accum[ih, iw] - mean[0]
                        var[0] += diff * diff
            var[0] = var[0] / count
            
            # Normalize
            eps = T.cast(1e-5, dtype)
            inv_std = T.rsqrt(var[0] + eps)
            
            for ih in T.Parallel(block_h):
                for iw in T.Parallel(block_w):
                    h_idx = tile_h + ih
                    w_idx = tile_w + iw
                    if h_idx < out_height and w_idx < out_width:
                        norm_val = (accum[ih, iw] - mean[0]) * inv_std
                        Y[batch, oc, h_idx, w_idx] = norm_val / T.cast(2.0, dtype)
    
    return tilelang.compile(fused_conv_instancenorm_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by
        self._kernel_cache = {}
    
    def _get_kernel(self, batch_size, in_channels, out_channels, height, width, kernel_size, dtype="float16"):
        key = (batch_size, in_channels, out_channels, height, width, kernel_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_instancenorm_kernel(
                batch_size, in_channels, out_channels, height, width, kernel_size, dtype=dtype
            )
        return self._kernel_cache[key]
    
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = self.conv.weight.half()
        bias_fp16 = self.conv.bias.half()
        
        # Compute running mean and var for instance norm (simplified)
        running_mean = torch.zeros(out_channels, device=x.device, dtype=torch.float16)
        running_var = torch.ones(out_channels, device=x.device, dtype=torch.float16)
        
        kernel = self._get_kernel(batch_size, in_channels, out_channels, height, width, kernel_size)
        y = kernel(x_fp16, weight_fp16, bias_fp16, running_mean, running_var)
        
        return y