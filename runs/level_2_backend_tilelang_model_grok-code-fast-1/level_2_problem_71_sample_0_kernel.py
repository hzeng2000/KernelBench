import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_div_leaky_kernel(M: int, N: int, divisor: float, negative_slope: float, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_div_leaky_kernel(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                val = A[y, x] / divisor
                C[y, x] = T.if_then_else(val > 0, val, negative_slope * val)

    return tilelang.compile(fused_div_leaky_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).half()
        self.divisor = divisor
        self.negative_slope = 0.01
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, divisor: float, negative_slope: float, tl_dtype: str):
        key = (M, N, divisor, negative_slope, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_div_leaky_kernel(M, N, divisor, negative_slope, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x.half())
        original_shape = x.shape
        
        x = x.view(-1, x.size(-1))

        M, N = x.shape
        kernel = self._get_kernel(M, N, self.divisor, self.negative_slope, "float16")
        x = kernel(x)

        return x.view(original_shape)