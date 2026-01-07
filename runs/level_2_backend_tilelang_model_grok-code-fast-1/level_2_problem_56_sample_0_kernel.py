import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_matmul_sigmoid_kernel(M: int, K: int, N: int, block_M: int = 16, block_N: int = 8, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_matmul_sigmoid_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((N,), dtype),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=False)
            
            for local_m in T.Parallel(block_M):
                for local_n in T.Parallel(block_N):
                    C_local[local_m, local_n] += C[by * block_N + local_n]
                    D[bx * block_M + local_m, by * block_N + local_n] = T.sigmoid(C_local[local_m, local_n])
    
    return tilelang.compile(fused_matmul_sigmoid_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication with sigmoid fused, then sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_sigmoid_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x_half = x.half().contiguous()
        weight_half = self.linear.weight.t().half().contiguous()
        bias_half = self.linear.bias.half().contiguous()
        
        batch_size, input_size = x.shape
        hidden_size = self.linear.out_features
        kernel = self._get_kernel(batch_size, input_size, hidden_size, "float16")
        D = kernel(x_half, weight_half, bias_half)
        
        x = torch.sum(D.float(), dim=1, keepdim=True)
        return x