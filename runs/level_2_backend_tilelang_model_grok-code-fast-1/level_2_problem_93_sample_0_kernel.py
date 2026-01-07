import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(M: int, N: int, add_value: float, multiply_value: float, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    temp = A[y, x] + add_value
                    temp = T.min(temp, T.cast(0.0, dtype))
                    # GELU approximation
                    gelu_arg = T.cast(0.7978845608028654, dtype) * (temp + T.cast(0.044715, dtype) * temp * temp * temp)
                    temp = T.cast(0.5, dtype) * temp * (T.cast(1.0, dtype) + T.tanh(gelu_arg))
                    C[y, x] = temp * multiply_value

    return tilelang.compile(fused_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then fuses the add, min, GELU, and multiply operations into a single TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, add_value: float, multiply_value: float, tl_dtype: str):
        key = (M, N, add_value, multiply_value, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(M, N, add_value, multiply_value, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x_c = x.contiguous().half()  # Convert to FP16
        original_shape = x_c.shape
        x_c = x_c.view(-1, x_c.size(-1))
        M, N = x_c.shape
        kernel = self._get_kernel(M, N, self.add_value, self.multiply_value, "float16")
        C = kernel(x_c)
        return C.view(original_shape).float()  # Convert back to FP32 for compatibility