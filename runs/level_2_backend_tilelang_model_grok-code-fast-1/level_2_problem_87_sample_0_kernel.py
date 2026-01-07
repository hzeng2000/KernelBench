import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_sub_mish_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_sub_mish_kernel(
        X: T.Tensor((M, N), dtype),
        a: T.float32,
        b: T.float32,
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    val = X[y, x] - T.cast(a, dtype) - T.cast(b, dtype)
                    sp = T.log(1 + T.exp(val))
                    t = T.tanh(sp)
                    Y[y, x] = val * t

    return tilelang.compile(fused_sub_mish_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_sub_mish_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x)
        x = x.to(torch.float16)
        
        # Get original shape for reshaping output
        original_shape = x.shape
        
        x = x.view(-1, x.size(-1))

        M, N = x.shape
        kernel = self._get_kernel(M, N, "float16")
        x = kernel(x, self.subtract_value_1, self.subtract_value_2)

        return x.view(original_shape).to(torch.float32)