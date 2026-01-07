import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(B: int, C: int, D: int, H: int, W: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    M = B * C
    N = D * H * W
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, N), dtype),
        Sum: T.Tensor((C,), dtype),
        Out: T.Tensor((M, N), dtype),
        channel_dim: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    val = A[y, x]
                    val = T.if_then_else(val > 0, val, T.cast(0.2, dtype) * val)
                    ch = T.mod(y, channel_dim)
                    val = val + Sum[ch]
                    val = T.max(T.min(val, T.cast(1.0, dtype)), T.cast(-1.0, dtype))
                    sqrt2 = T.sqrt(T.cast(2.0, dtype))
                    erf_arg = val / sqrt2
                    gelu_val = T.cast(0.5, dtype) * val * (T.cast(1.0, dtype) + T.erf(erf_arg))
                    Out[y, x] = gelu_val

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, then fuses LeakyReLU, addition with sum_tensor, clamp, and GELU into a single TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = (B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x)
        x_c = x.contiguous().half()
        sum_t = self.sum_tensor.contiguous().half()
        
        B, C, D, H, W = x_c.shape
        M = B * C
        N = D * H * W
        
        x_c = x_c.view(M, N)
        sum_t = sum_t.view(C)
        
        kernel = self._get_kernel(B, C, D, H, W, "float16")
        out = kernel(x_c, sum_t, channel_dim=C)
        
        out = out.view(B, C, D, H, W).float()
        return out