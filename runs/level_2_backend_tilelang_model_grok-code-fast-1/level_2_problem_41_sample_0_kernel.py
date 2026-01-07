import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias_gemm: T.Tensor((N,), dtype),
        gamma: T.Tensor((N,), dtype),
        beta: T.Tensor((N,), dtype),
        running_mean: T.Tensor((N,), dtype),
        running_var: T.Tensor((N,), dtype),
        eps: T.float32,
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N:
                    x = C_local[i, j] + bias_gemm[n]
                    x = gamma[n] * (x - running_mean[n]) / T.sqrt(running_var[n] + eps) + beta[n]
                    # GELU approximation
                    sqrt_2_pi = T.sqrt(2.0 / 3.141592653589793)
                    x = 0.5 * x * (1.0 + T.tanh(sqrt_2_pi * (x + 0.044715 * x * x * x)))
                    # ReLU
                    x = T.max(x, 0.0)
                    C_local[i, j] = x
            
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tilelang.compile(fused_kernel, out_idx=[8], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs GEMM, BatchNorm, GELU, and ReLU in a fused TileLang kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = x.contiguous().to(torch.float16)
        A = x.view(-1, x.size(-1))
        B = self.gemm.weight.contiguous().to(torch.float16)
        bias_gemm = self.gemm.bias.contiguous().to(torch.float16)
        gamma = self.batch_norm.weight.contiguous().to(torch.float16)
        beta = self.batch_norm.bias.contiguous().to(torch.float16)
        running_mean = self.batch_norm.running_mean.contiguous().to(torch.float16)
        running_var = self.batch_norm.running_var.contiguous().to(torch.float16)
        eps = self.batch_norm.eps
        
        M, K = A.shape
        N = B.shape[0]
        kernel = self._get_kernel(M, K, N, "float16")
        C = torch.empty(M, N, dtype=torch.float16, device=x.device)
        kernel(A, B, bias_gemm, gamma, beta, running_mean, running_var, eps, C)
        
        return C.view(x.shape[0], -1)