import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_kernel(batch: int, in_features: int, out_features: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        scaling_factor: T.float32,
        hardtanh_min: T.float32,
        hardtanh_max: T.float32,
        Y: T.Tensor((batch, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(out_features, block_N), T.ceildiv(batch, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            for k in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=3):
                for i, j in T.Parallel(block_M, block_K):
                    A_shared[i, j] = X[by * block_M + i, k * block_K + j]
                for i, j in T.Parallel(block_K, block_N):
                    B_shared[i, j] = W[bx * block_N + j, k * block_K + i]
                for i, j, kk in T.Parallel(block_M, block_N, block_K):
                    if k == 0:
                        C_local[i, j] = 0.0
                    C_local[i, j] += A_shared[i, kk] * B_shared[kk, j]

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += B[bx * block_N + j]
                C_local[i, j] *= scaling_factor
                C_local[i, j] = T.max(T.min(C_local[i, j], hardtanh_max), hardtanh_min)
                x = C_local[i, j]
                gelu_arg = 0.7978845608028654 * (x + 0.044715 * x * x * x)
                C_local[i, j] = 0.5 * x * (1.0 + T.tanh(gelu_arg))

            for i, j in T.Parallel(block_M, block_N):
                Y[by * block_M + i, bx * block_N + j] = C_local[i, j]

    return tilelang.compile(fused_kernel, out_idx=[6], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that fuses GEMM, scaling, hardtanh, and GELU into a single TileLang kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, in_f: int, out_f: int, tl_dtype: str):
        key = (batch, in_f, out_f, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(batch, in_f, out_f, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.contiguous()
        batch, in_f = x.shape
        out_f = self.out_features
        kernel = self._get_kernel(batch, in_f, out_f, "float16")
        y = kernel(x, self.weight, self.bias, self.scaling_factor, self.hardtanh_min, self.hardtanh_max)
        return y