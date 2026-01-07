import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, ko * block_K : (ko + 1) * block_K], A_shared)
                T.copy(B[ko * block_K : (ko + 1) * block_K, bx * block_N : (bx + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(block_M, block_N):
                c_val = C_local[i, j] + bias[bx * block_N + j]
                c_val = c_val * (T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-c_val)))
                C_local[i, j] = c_val
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
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
        A = x.contiguous().half()
        B = self.matmul.weight.t().contiguous().half()
        bias = self.bias.contiguous().half()
        
        M, K = A.shape
        N = B.shape[1]
        kernel = self._get_kernel(M, K, N, "float16")
        C = kernel(A, B, bias)
        
        x = self.group_norm(C.float()).half()
        return x