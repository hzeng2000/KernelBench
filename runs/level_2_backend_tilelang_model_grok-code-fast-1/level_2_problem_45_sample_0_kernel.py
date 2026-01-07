import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_linear_sigmoid_kernel(M: int, K: int, N: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def linear_sigmoid_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                for i, j, kk in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, kk] * B_shared[j, kk]
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += bias[bx * block_N + j]
                C_local[i, j] = 1.0 / (1.0 + T.exp(-C_local[i, j]))
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tilelang.compile(linear_sigmoid_kernel, out_idx=[3], target="cuda")


def build_linear_kernel(M: int, K: int, N: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def linear_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                for i, j, kk in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, kk] * B_shared[j, kk]
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += bias[bx * block_N + j]
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tilelang.compile(linear_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication (Gemm) with fused Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self._kernel_cache1 = {}
        self._kernel_cache2 = {}

    def _get_kernel1(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache1:
            self._kernel_cache1[key] = build_linear_sigmoid_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache1[key]

    def _get_kernel2(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache2:
            self._kernel_cache2[key] = build_linear_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache2[key]

    def forward(self, x):
        x = x.half()
        x = self._get_kernel1(x.shape[0], x.shape[1], self.linear1.out_features, "float16")(
            x, self.linear1.weight.t().half(), self.linear1.bias.half()
        )
        x = self._get_kernel2(x.shape[0], x.shape[1], self.linear2.out_features, "float16")(
            x, self.linear2.weight.t().half(), self.linear2.bias.half()
        )
        x = x.float()
        x = torch.logsumexp(x, dim=1)
        return x