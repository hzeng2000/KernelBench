import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemm_bias_relu_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_bias_relu_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            acc = T.alloc_fragment((block_M, block_N), dtype)
            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] = T.cast(0, dtype)

            for k in T.serial(T.ceildiv(K, block_K)):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)

                for i, j in T.Parallel(block_M, block_K):
                    if start_y + i < M and k * block_K + j < K:
                        A_shared[i, j] = A[start_y + i, k * block_K + j]
                    else:
                        A_shared[i, j] = T.cast(0, dtype)

                for i, j in T.Parallel(block_K, block_N):
                    if k * block_K + i < K and start_x + j < N:
                        B_shared[i, j] = B[k * block_K + i, start_x + j]
                    else:
                        B_shared[i, j] = T.cast(0, dtype)

                for i, j in T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        acc[i, j] += A_shared[i, kk] * B_shared[kk, j]

            for i, j in T.Parallel(block_M, block_N):
                if start_y + i < M and start_x + j < N:
                    C[start_y + i, start_x + j] = T.max(acc[i, j] + bias[start_x + j], T.cast(0, dtype))

    return tilelang.compile(gemm_bias_relu_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_bias_relu_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        
        M, K = x_c.shape
        N = self.out_features
        
        kernel = self._get_kernel(M, N, K, "float16")
        
        weight_t = self.weight.t().contiguous()
        bias_c = self.bias.contiguous()
        
        output = kernel(x_c, weight_t, bias_c)
        
        return output