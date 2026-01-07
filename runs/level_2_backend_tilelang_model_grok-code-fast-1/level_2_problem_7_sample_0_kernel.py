import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_activation_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_activation_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    temp = A[y, x]
                    # ReLU
                    temp = T.max(temp, T.cast(0.0, dtype))
                    # LeakyReLU
                    temp = T.if_then_else(temp > 0, temp, temp * T.cast(0.01, dtype))
                    # GELU approximation
                    sqrt_2_pi = T.sqrt(T.cast(2.0 / math.pi, dtype))
                    temp_cubed = temp * temp * temp
                    inner = sqrt_2_pi * (temp + T.cast(0.044715, dtype) * temp_cubed)
                    temp = T.cast(0.5, dtype) * temp * (T.cast(1.0, dtype) + T.tanh(inner))
                    # Sigmoid
                    temp = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-temp))
                    # Add bias
                    temp = temp + B[x]
                    C[y, x] = temp

    return tilelang.compile(fused_activation_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence using fused TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_activation_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x).half()  # Convert to FP16
        original_shape = x.shape
        
        x = x.view(-1, x.size(-1))
        bias = self.bias.view(-1).half()

        M, N = x.shape
        kernel = self._get_kernel(M, N, "float16")
        x = kernel(x, bias)

        return x.view(original_shape)