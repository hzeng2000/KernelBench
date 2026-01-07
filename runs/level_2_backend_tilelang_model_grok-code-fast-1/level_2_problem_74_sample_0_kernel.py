import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_leaky_mul_leaky_kernel(B: int, C: int, D: int, H: int, W: int, block_D: int = 4, block_H: int = 8, block_W: int = 8, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_leaky_mul_leaky_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        Mul: T.Tensor((C, 1, 1, 1), dtype),
        C_out: T.Tensor((B, C, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(D, block_D), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bd, bh, bw):
            start_d = bd * block_D
            start_h = bh * block_H
            start_w = bw * block_W

            for b in T.serial(B):
                for c in T.serial(C):
                    for local_d, local_h, local_w in T.Parallel(block_D, block_H, block_W):
                        d = start_d + local_d
                        h = start_h + local_h
                        w = start_w + local_w

                        if d < D and h < H and w < W:
                            val = A[b, c, d, h, w]
                            val = T.max(val, val * 0.2)  # LeakyReLU with negative_slope=0.2
                            val = val * Mul[c, 0, 0, 0]
                            val = T.max(val, val * 0.2)  # LeakyReLU again
                            C_out[b, c, d, h, w] = val

    return tilelang.compile(fused_leaky_mul_leaky_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies a fused LeakyReLU + multiplication + LeakyReLU kernel, 
    and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = (B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_leaky_mul_leaky_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        B, C, D, H, W = x.shape
        kernel = self._get_kernel(B, C, D, H, W, "float16")
        x = kernel(x.half(), self.multiplier.half()).float()
        x = self.max_pool(x)
        return x