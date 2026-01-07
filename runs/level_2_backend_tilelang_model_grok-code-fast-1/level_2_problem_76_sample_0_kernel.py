import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemm_bias_relu_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 64, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_bias_relu_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((M,), dtype),
        C: T.Tensor((N, M), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                
                with T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        C_local[T.thread_y, T.thread_x] += A_shared[T.thread_y, kk] * B_shared[T.thread_x, kk]
            
            with T.Parallel(block_M, block_N):
                m = bx * block_M + T.thread_y
                n = by * block_N + T.thread_x
                if m < M and n < N:
                    C[n, m] = T.max(C_local[T.thread_y, T.thread_x] + bias[m], T.float16(0))
    
    return tilelang.compile(gemm_bias_relu_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, bias add, and ReLU into a single TileLang kernel.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_bias_relu_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        x = x.to(torch.float16)
        weight = self.weight.to(torch.float16)
        bias = self.bias.to(torch.float16)
        
        N, K = x.shape
        M = weight.shape[0]
        
        kernel = self._get_kernel(M, N, K, "float16")
        C = kernel(weight, x, bias)
        
        return C