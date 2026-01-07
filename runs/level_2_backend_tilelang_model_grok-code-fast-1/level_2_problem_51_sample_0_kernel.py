import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_matmul_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),
        subtract: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[k * block_K : (k + 1) * block_K, bx * block_N : (bx + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += bias[bx * block_N + j] - subtract[bx * block_N + j]
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tilelang.compile(matmul_kernel, out_idx=[4], target="cuda")


def build_lse_gelu_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def lse_gelu_kernel(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            max_vals = T.alloc_fragment((block_M,), dtype)
            sum_vals = T.alloc_fragment((block_M,), dtype)
            T.fill(max_vals, -float("inf"))
            T.fill(sum_vals, 0.0)
            for k in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_N : (k + 1) * block_N], A_shared)
                for i in T.Parallel(block_M):
                    for j in T.serial(block_N):
                        max_vals[i] = T.max(max_vals[i], A_shared[i, j])
            for k in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_N : (k + 1) * block_N], A_shared)
                for i in T.Parallel(block_M):
                    for j in T.serial(block_N):
                        sum_vals[i] += T.exp(A_shared[i, j] - max_vals[i])
            for i in T.Parallel(block_M):
                lse = T.log(sum_vals[i]) + max_vals[i]
                x = lse
                gelu_x = 0.5 * x * (1 + T.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
                C[bx * block_M + i, 0] = gelu_x

    return tilelang.compile(lse_gelu_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.subtract = nn.Parameter(torch.randn(out_features))
        self._kernel_cache_matmul = {}
        self._kernel_cache_lse = {}

    def _get_matmul_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache_matmul:
            self._kernel_cache_matmul[key] = build_matmul_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache_matmul[key]

    def _get_lse_gelu_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache_lse:
            self._kernel_cache_lse[key] = build_lse_gelu_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache_lse[key]

    def forward(self, x):
        original_x = x.clone().detach()
        
        A_c = x.contiguous()
        B_c = self.gemm_weight.t().contiguous()
        bias_c = self.gemm_bias.contiguous() if self.gemm_bias is not None else torch.zeros_like(self.subtract)
        subtract_c = self.subtract.contiguous()
        
        M, K = A_c.shape
        N = B_c.shape[1]
        kernel_matmul = self._get_matmul_kernel(M, K, N, "float16")
        C = kernel_matmul(A_c, B_c, bias_c, subtract_c)
        
        M_c, N_c = C.shape
        kernel_lse_gelu = self._get_lse_gelu_kernel(M_c, N_c, "float16")
        x = kernel_lse_gelu(C)
        
        return x + original_x