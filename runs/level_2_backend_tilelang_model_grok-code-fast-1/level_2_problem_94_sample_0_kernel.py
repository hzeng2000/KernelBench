import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_gemm_bias_hardtanh_mish_kernel(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_gemm_bias_hardtanh_mish_kernel(
        A: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
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
                temp = C_local[i, j] + bias[bx * block_N + j]
                temp = T.clamp(temp, T.float16(-1.0), T.float16(1.0))  # Hardtanh
                # Mish: x * tanh(softplus(x)), softplus(x) = log(exp(x) + 1)
                softplus_temp = T.log(T.exp(temp) + T.float16(1.0))
                temp = temp * T.tanh(softplus_temp)
                C[by * block_M + i, bx * block_N + j] = temp
    
    return tilelang.compile(fused_gemm_bias_hardtanh_mish_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + BiasAdd + Hardtanh + Mish in a single TileLang kernel.
    GroupNorm remains as PyTorch operator.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_gemm_bias_hardtanh_mish_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = x.contiguous()
        W = self.gemm.weight.contiguous()
        bias = self.bias.contiguous()
        
        M, K = x.shape
        N = W.shape[0]
        
        kernel = self._get_kernel(M, N, K, "float16")
        x = kernel(x, W, bias)
        
        x = self.groupnorm(x)
        return x