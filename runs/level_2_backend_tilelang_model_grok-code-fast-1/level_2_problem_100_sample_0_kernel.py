import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_clamp_divide_kernel(M: int, N: int, min_value: float, divisor: float, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def clamp_divide_kernel(
        A: T.Tensor((M, N), dtype),
        min_val: T.float32,
        div: T.float32,
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    C[y, x] = T.max(A[y, x], min_val) / div

    return tilelang.compile(clamp_divide_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_clamp_divide_kernel(M, N, self.min_value, self.divisor, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.half()  # Convert to FP16 for TileLang kernel
        original_shape = x.shape
        x = x.view(-1, x.size(-1))
        M, N = x.shape
        kernel = self._get_kernel(M, N, "float16")
        x = kernel(x, self.min_value, self.divisor)
        return x.view(original_shape)