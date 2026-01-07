import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_kernel(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_linear_kernel(
        A: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        S: T.Tensor((), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(W[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], W_shared)
                
                for i, j, kk in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, kk] * W_shared[j, kk]
            
            for i, j in T.Parallel(block_M, block_N):
                temp = C_local[i, j] + B[bx * block_N + j]
                C[by * block_M + i, bx * block_N + j] = temp * S[()] + temp

    return tilelang.compile(fused_linear_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16).contiguous()
        M, K = x.shape
        N = self.weight.shape[0]
        kernel = self._get_kernel(M, N, K, "float16")
        S = torch.tensor(self.scaling_factor, dtype=torch.float16, device=x.device)
        C = kernel(x, self.weight, self.bias, S)
        return C