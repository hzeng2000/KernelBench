import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemm_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 64, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            start_x = bx * block_N
            start_y = by * block_M

            T.clear(C_local)

            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[start_y: start_y + block_M, k * block_K: (k + 1) * block_K], A_shared)
                T.copy(B[start_x: start_x + block_N, k * block_K: (k + 1) * block_K], B_shared)
                for i, j, p in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, p] * B_shared[j, p]

            for i, j in T.Parallel(block_M, block_N):
                C[start_y + i, start_x + j] = C_local[i, j] + bias[start_x + j]

    return tilelang.compile(gemm_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that performs a GEMM with custom TileLang kernel, applies Group Normalization, and then HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self._kernel_cache = {}
        self.in_features = in_features
        self.out_features = out_features

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x_c = x.contiguous()
        weight_c = self.gemm.weight.contiguous()
        bias_c = self.gemm.bias.contiguous()
        
        # Get shapes
        M, K = x_c.shape
        N = self.out_features
        
        kernel = self._get_kernel(M, N, K, "float16")
        C = kernel(x_c, weight_c, bias_c)
        
        C = self.group_norm(C)
        C = self.hardtanh(C)
        return C