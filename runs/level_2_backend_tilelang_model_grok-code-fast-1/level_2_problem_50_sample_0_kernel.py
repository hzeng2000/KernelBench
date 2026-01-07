import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_bias_scale_kernel(M: int, N: int, C: int, D: int, H: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_bias_scale_kernel(
        A: T.Tensor((M, N), dtype),
        Bias: T.Tensor((C,), dtype),
        Scale: T.float32,
        C_out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    # Compute c: y = b*(C*D*H) + c*(D*H) + d*H + h
                    c = (y // (D * H)) % C
                    C_out[y, x] = (A[y, x] + Bias[c]) * Scale

    return tilelang.compile(fused_bias_scale_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, average pooling, fused bias addition and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Absorb scale1 into conv weights
        self.conv_transpose.weight.data *= scale1
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = scale2
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, C: int, D: int, H: int, tl_dtype: str):
        key = (M, N, C, D, H, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_bias_scale_kernel(M, N, C, D, H, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.avg_pool(x)
        # Get shapes
        B, C, D, H, W = x.shape
        M = B * C * D * H
        N = H * W  # Note: H and W are the same after avgpool, but H is height, W is width, N = height * width
        # Actually, after avgpool, shape is (B, C, D_out, H_out, W_out), with H_out = W_out = 31, D_out=15
        # So N = H_out * W_out = 31 * 31 = 961
        x_c = x.contiguous().view(M, N)
        bias_c = self.bias.contiguous().view(-1)  # (C,)
        kernel = self._get_kernel(M, N, C, D, H, "float16")
        x_out = kernel(x_c, bias_c, self.scale2)
        return x_out.view(B, C, D, H, W)