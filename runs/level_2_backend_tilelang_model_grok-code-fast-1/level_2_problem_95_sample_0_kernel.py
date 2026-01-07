import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_gemm_add_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, num_stages: int = 3, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_add_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        add_value: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += bias[bx * block_N + j] + add_value[bx * block_N + j]

            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tilelang.compile(gemm_add_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that fuses matrix multiplication, bias addition, and add_value addition into a single TileLang kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_add_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x_c = x.contiguous()
        weight_c = self.matmul.weight.contiguous()
        bias_c = self.matmul.bias.contiguous()
        add_value_c = self.add_value.contiguous()
        
        M, K = x_c.shape
        N = weight_c.shape[0]
        
        kernel = self._get_kernel(M, K, N, "float16")
        x = kernel(x_c.to(torch.float16), weight_c.to(torch.float16), bias_c.to(torch.float16), add_value_c.to(torch.float16)).to(x.dtype)
        
        x = torch.sigmoid(x) * x  # Swish
        x = torch.tanh(x)
        x = torch.nn.functional.gelu(x)  # GELU
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1)  # Hardtanh
        return x