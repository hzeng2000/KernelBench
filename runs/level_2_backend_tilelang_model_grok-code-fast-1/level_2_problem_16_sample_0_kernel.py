import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_post_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_post_kernel(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        add_value: T.float32,
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                a = A[y, x]
                # Mish: a * tanh(softplus(a)) where softplus(a) = log(1 + exp(a))
                softplus_a = T.log(1 + T.exp(a))
                mish_a = a * T.tanh(softplus_a)
                # Add value
                temp = mish_a + add_value
                # Hardtanh: clamp to [-1, 1]
                clamped = T.min(T.max(temp, T.float16(-1)), T.float16(1))
                # Scale
                C[y, x] = clamped * T.float16(scale)

    return tilelang.compile(fused_post_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution in FP16, then applies fused Mish + add + Hardtanh + scale in a custom TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding).half()
        self.add_value = add_value
        self.scale = scale
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_post_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        
        # Get original shape for reshaping output
        original_shape = x.shape
        
        x = x.view(-1, x.size(-1))

        M, N = x.shape
        kernel = self._get_kernel(M, N, "float16")
        x = kernel(x, self.add_value, self.scale)

        return x.view(original_shape)