import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv_gn_tanh_hardswish_kernel(
    batch_size: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    groups: int,
    block_size: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_height = height
    out_width = width
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
        Weight: T.Tensor((out_channels, out_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Scale: T.Tensor((out_channels,), dtype),
        BiasGN: T.Tensor((out_channels,), dtype),
        Out: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_height, block_size), T.ceildiv(out_width, block_size), batch_size, threads=threads) as (by, bx, bz):
            start_y = by * block_size
            start_x = bx * block_size
            batch_idx = bz

            for local_y, local_x in T.Parallel(block_size, block_size):
                y = start_y + local_y
                x = start_x + local_x

                if y < out_height and x < out_width:
                    for c in range(out_channels):
                        acc = 0.0
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                h_in = y + kh - pad
                                w_in = x + kw - pad
                                if h_in >= 0 and h_in < height and w_in >= 0 and w_in < width:
                                    acc += X[batch_idx, c, h_in, w_in] * Weight[c, c, kh, kw]
                        acc += Bias[c]
                        
                        # GroupNorm
                        group_size = out_channels // groups
                        group_idx = c // group_size
                        mean = 0.0
                        var = 0.0
                        for g in range(group_size):
                            gc = group_idx * group_size + g
                            mean += X[batch_idx, gc, y, x]
                        mean /= group_size
                        for g in range(group_size):
                            gc = group_idx * group_size + g
                            diff = X[batch_idx, gc, y, x] - mean
                            var += diff * diff
                        var /= group_size
                        std = T.sqrt(var + 1e-5)
                        norm_val = (acc - mean) / std
                        norm_val = norm_val * Scale[c] + BiasGN[c]
                        
                        # Tanh
                        tanh_val = T.tanh(norm_val)
                        
                        # HardSwish
                        hard_swish_val = tanh_val * T.max(0.0, T.min(1.0, (norm_val + 3.0) / 6.0))
                        
                        Out[batch_idx, c, y, x] = hard_swish_val

    return tilelang.compile(fused_kernel, out_idx=[5], target="cuda")


def build_residual_add_logsumexp_kernel(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    block_size: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def residual_kernel(
        A: T.Tensor((batch_size, channels, height, width), dtype),
        B: T.Tensor((batch_size, channels, height, width), dtype),
        Out: T.Tensor((batch_size, 1, height, width), dtype),
    ):
        with T.Kernel(T.ceildiv(height, block_size), T.ceildiv(width, block_size), batch_size, threads=threads) as (by, bx, bz):
            start_y = by * block_size
            start_x = bx * block_size
            batch_idx = bz

            for local_y, local_x in T.Parallel(block_size, block_size):
                y = start_y + local_y
                x = start_x + local_x

                if y < height and x < width:
                    max_val = -1e10
                    for c in range(channels):
                        val = A[batch_idx, c, y, x] + B[batch_idx, c, y, x]
                        if val > max_val:
                            max_val = val
                    
                    sum_exp = 0.0
                    for c in range(channels):
                        val = A[batch_idx, c, y, x] + B[batch_idx, c, y, x]
                        sum_exp += T.exp(val - max_val)
                    
                    Out[batch_idx, 0, y, x] = T.log(sum_exp) + max_val

    return tilelang.compile(residual_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        
        self._fused_kernel_cache = {}
        self._residual_kernel_cache = {}
        
    def _get_fused_kernel(self, batch_size: int, out_channels: int, height: int, width: int, kernel_size: int, groups: int):
        key = (batch_size, out_channels, height, width, kernel_size, groups)
        if key not in self._fused_kernel_cache:
            self._fused_kernel_cache[key] = build_fused_conv_gn_tanh_hardswish_kernel(
                batch_size, out_channels, height, width, kernel_size, groups
            )
        return self._fused_kernel_cache[key]
    
    def _get_residual_kernel(self, batch_size: int, channels: int, height: int, width: int):
        key = (batch_size, channels, height, width)
        if key not in self._residual_kernel_cache:
            self._residual_kernel_cache[key] = build_residual_add_logsumexp_kernel(
                batch_size, channels, height, width
            )
        return self._residual_kernel_cache[key]

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        
        # Convolution
        x_conv = self.conv(x.half())
        
        # Get GroupNorm parameters
        scale = self.group_norm.weight.half()
        bias_gn = self.group_norm.bias.half()
        
        # Fused GN + Tanh + HardSwish
        kernel = self._get_fused_kernel(batch_size, out_channels, height, width, self.conv.kernel_size[0], self.group_norm.num_groups)
        x_fused = kernel(x_conv, self.conv.weight.half(), self.conv.bias.half(), scale, bias_gn)
        
        # Residual Add + LogSumExp
        residual_kernel = self._get_residual_kernel(batch_size, out_channels, height, width)
        x_logsumexp = residual_kernel(x_conv, x_fused)
        
        return x_logsumexp.float()