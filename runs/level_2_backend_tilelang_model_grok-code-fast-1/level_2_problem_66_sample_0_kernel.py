import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(M: int, K: int, N: int, block_M: int = 128, block_K: int = 128, block_N: int = 128, threads: int = 128, num_stages: int = 3, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Mask: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M :, k * block_K :], A_shared)
                T.copy(W[bx * block_N :, k * block_K :], W_shared, transpose=True)
                T.gemm(A_shared, W_shared, C_local)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = (C_local[i, j] + B[bx * block_N + j]) * Mask[by * block_M + i, bx * block_N + j]
            T.copy(C_local, C[by * block_M :, bx * block_N :])

    return tilelang.compile(fused_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        self.b = nn.Parameter(torch.empty(out_features))
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        A = x.contiguous().half()
        W = self.W.contiguous().half()
        B = self.b.contiguous().half()
        Mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout_p)).half() / (1 - self.dropout_p)
        M, K = A.shape
        N = self.out_features
        kernel = self._get_kernel(M, K, N, "float16")
        C = kernel(A, W, B, Mask)
        return torch.softmax(C, dim=1)