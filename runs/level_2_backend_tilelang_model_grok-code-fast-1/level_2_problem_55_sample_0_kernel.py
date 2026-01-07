import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemm_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[k * block_K : (k + 1) * block_K, bx * block_N : (bx + 1) * block_N], B_shared)
                for i, j in T.Parallel(block_M, block_N):
                    for l in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, l] * B_shared[l, j]
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tilelang.compile(gemm_kernel, out_idx=[2], target="cuda")


def build_reduction_kernel(B: int, O: int, kernel_size: int = 2, scale_factor: float = 0.5, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def reduction_kernel(
        C: T.Tensor((B, O), dtype),
        Out: T.Tensor((B,), dtype),
    ):
        with T.Kernel(B, threads=threads) as bx:
            sum_val = T.alloc_fragment((), dtype)
            T.clear(sum_val)
            for i in T.Serial(O // kernel_size):
                max_val = T.max(C[bx, i * kernel_size], C[bx, i * kernel_size + 1])
                sum_val[()] += max_val
            Out[bx] = sum_val[()] * scale_factor

    return tilelang.compile(reduction_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs matrix multiplication, max pooling, sum, and scaling using TileLang kernels.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self._kernel_cache = {}

    def _get_gemm_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = ("gemm", M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_reduction_kernel(self, B: int, O: int, kernel_size: int, scale_factor: float, tl_dtype: str):
        key = ("reduction", B, O, kernel_size, scale_factor, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_reduction_kernel(B, O, kernel_size, scale_factor, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x_c = x.contiguous().to(torch.float16)
        weight_c = self.matmul.weight.t().contiguous().to(torch.float16)
        
        batch_size, in_features = x_c.shape
        out_features = self.matmul.out_features
        
        gemm_kernel = self._get_gemm_kernel(batch_size, in_features, out_features, "float16")
        C = gemm_kernel(x_c, weight_c)
        
        reduction_kernel = self._get_reduction_kernel(batch_size, out_features, self.kernel_size, self.scale_factor, "float16")
        out = reduction_kernel(C)
        
        return out.to(torch.float32)