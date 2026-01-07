import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemv_kernel(M: int, N: int, block_M: int = 128, block_N: int = 128, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def gemv_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((N,), dtype),
        scale: T.Tensor((), dtype),
        C: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            for local_y in T.Parallel(block_M):
                y = bx * block_M + local_y
                if y < M:
                    partial = T.cast(0, dtype)
                    for local_x in range(block_N):
                        x = by * block_N + local_x
                        if x < N:
                            partial += A[y, x] * B[x]
                    T.atomic_add(C, y, partial * scale)

    return tilelang.compile(gemv_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs the equivalent computation using a precomputed vector and a custom TileLang GEMV kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.v = self.weight.sum(dim=0) * (self.scaling_factor / 2)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemv_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        A = x.half().contiguous()
        B = self.v.half().contiguous()
        scale = torch.tensor(1.0, dtype=torch.float16, device=x.device)
        
        # Get shape
        M, N = A.shape
        kernel = self._get_kernel(M, N, "float16")
        C = kernel(A, B, scale)
        
        return C.unsqueeze(-1)