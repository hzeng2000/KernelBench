import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
from tilelang import tvm as tl_tvm
import math


def build_logsumexp_kernel(B: int, C: int, D: int, H: int, W: int, block_B: int = 1, block_D: int = 8, block_H: int = 8, block_W: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def logsumexp_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        B_out: T.Tensor((B, 1, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(D, block_D), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bb, bd, bh, bw):
            for local_b, local_d, local_h, local_w in T.Parallel(block_B, block_D, block_H, block_W):
                b = bb * block_B + local_b
                d = bd * block_D + local_d
                h = bh * block_H + local_h
                w = bw * block_W + local_w
                
                if b < B and d < D and h < H and w < W:
                    max_val = T.float32(-tl_tvm.FInf())
                    for c in range(C):
                        max_val = T.max(max_val, T.cast(A[b, c, d, h, w], "float32"))
                    
                    sum_exp = T.float32(0.0)
                    for c in range(C):
                        sum_exp = sum_exp + T.exp(T.cast(A[b, c, d, h, w], "float32") - max_val)
                    
                    B_out[b, 0, d, h, w] = T.cast(T.log(sum_exp) + max_val, dtype)

    return tilelang.compile(logsumexp_kernel, out_idx=[1], target="cuda")


def build_fused_hardswish_sub_clamp_kernel(B: int, D: int, H: int, W: int, block_B: int = 1, block_D: int = 8, block_H: int = 8, block_W: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_hardswish_sub_clamp_kernel(
        A: T.Tensor((B, 1, D, H, W), dtype),
        bias: T.Tensor((1, 1, 1, 1), dtype),
        C: T.Tensor((B, 1, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(D, block_D), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bb, bd, bh, bw):
            for local_b, local_d, local_h, local_w in T.Parallel(block_B, block_D, block_H, block_W):
                b = bb * block_B + local_b
                d = bd * block_D + local_d
                h = bh * block_H + local_h
                w = bw * block_W + local_w
                
                if b < B and d < D and h < H and w < W:
                    x = T.cast(A[b, 0, d, h, w], "float32")
                    y = x + 3.0
                    sig = 1.0 / (1.0 + T.exp(-y))
                    hs = x * sig / 6.0
                    hs_sub = hs - T.cast(bias[0, 0, 0, 0], "float32")
                    clamped = T.max(T.min(hs_sub, 1.0), -1.0)
                    C[b, 0, d, h, w] = T.cast(clamped, dtype)

    return tilelang.compile(fused_hardswish_sub_clamp_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, custom LogSumExp, fused HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1)) 
        self._kernel_cache = {}

    def _get_logsumexp_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = ("logsumexp", B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_logsumexp_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_fused_kernel(self, B: int, D: int, H: int, W: int, tl_dtype: str):
        key = ("fused", B, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_hardswish_sub_clamp_kernel(B, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half()
        x = self.conv_transpose(x)
        # x shape: (128, 16, 31, 63, 63)
        B, C, D, H, W = x.shape
        logsumexp_kernel = self._get_logsumexp_kernel(B, C, D, H, W, "float16")
        x_logsumexp = logsumexp_kernel(x)
        # x_logsumexp shape: (128, 1, 31, 63, 63)
        fused_kernel = self._get_fused_kernel(B, D, H, W, "float16")
        bias_half = self.bias.half()
        x_out = fused_kernel(x_logsumexp, bias_half)
        return x_out.float()