import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, N), dtype),
        Mul: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    val = A[y, x] * Mul[x]
                    # LeakyReLU: max(val, 0.01 * val)
                    val = T.max(val, 0.01 * val)
                    # GELU: 0.5 * val * (1 + tanh(sqrt(2/pi) * (val + 0.044715 * val^3)))
                    sqrt_2_pi = T.sqrt(2.0 / math.pi)
                    val_cubed = val * val * val
                    tanh_arg = sqrt_2_pi * (val + 0.044715 * val_cubed)
                    gelu_val = 0.5 * val * (1.0 + T.tanh(tanh_arg))
                    C[y, x] = gelu_val

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, then fuses multiplication by learnable scalar, LeakyReLU, and GELU into a single TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x)
        x_c = x.contiguous()
        
        # Get original shape for reshaping output
        original_shape = x_c.shape
        
        # Reshape to (B*H*W, C)
        x_c = x_c.view(-1, x_c.size(1))
        mul_c = self.multiplier.view(-1).contiguous()

        M, N = x_c.shape
        kernel = self._get_kernel(M, N, "float16")
        y = kernel(x_c, mul_c)

        return y.view(original_shape)