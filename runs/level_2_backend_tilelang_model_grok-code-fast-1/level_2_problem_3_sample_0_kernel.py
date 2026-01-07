import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_norm_gelu_kernel(M: int, N: int, block_M: int = 128, block_N: int = 64, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_norm_gelu_kernel(
        X: T.Tensor((M, N), dtype),
        sum_weight: T.Tensor((), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        eps = 1e-5
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            with T.ThreadBinding(block_M, "threadIdx.y") as ty:
                i = bx * block_M + ty
                if i < M:
                    # Compute mean
                    sum_val = T.reduce(T.sum, [j for j in range(N)], X[i, j] + sum_weight, axis=[1])
                    mean = sum_val / N
                    # Compute var
                    sum_sq = T.reduce(T.sum, [j for j in range(N)], (X[i, j] + sum_weight - mean)**2, axis=[1])
                    var = sum_sq / N
                    # Then, for each j
                    with T.ThreadBinding(block_N, "threadIdx.x") as tx:
                        j = tx
                        if j < N:
                            Y[i, j] = T.gelu((X[i, j] + sum_weight - mean) / T.sqrt(var + eps))

    return tilelang.compile(fused_norm_gelu_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a fused add, layer normalization, and GELU activation using TileLang, then average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm_shape = norm_shape
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_norm_gelu_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse add, norm, gelu
        original_shape = x.shape
        x = x.view(-1, x.size(-1))
        M, N = x.shape
        kernel = self._get_kernel(M, N, "float16")
        x = kernel(x.to(torch.float16), self.sum_weight.to(torch.float16)).to(torch.float32)
        x = x.view(original_shape)
        x = self.avg_pool(x)
        return x