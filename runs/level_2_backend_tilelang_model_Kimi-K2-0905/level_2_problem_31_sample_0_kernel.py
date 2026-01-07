import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_conv_min_add_scale_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    constant_value: float,
    scaling_factor: float,
    block_M: int = 8,
    block_N: int = 32,
    block_K: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_height = height
    out_width = width

    @T.prim_func
    def fused_conv_min_add_scale(
        Input: T.Tensor((batch_size, in_channels, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels, 1, 1), dtype),
        Output: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_N),
            T.ceildiv(out_height, block_M),
            batch_size * out_channels,
            threads=threads
        ) as (bx, by, bz):
            tile_x = bx * block_N
            tile_y = by * block_M
            n = bz // batch_size
            b = bz % batch_size

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = tile_y + local_y
                x = tile_x + local_x

                if y < out_height and x < out_width:
                    acc = T.alloc_fragment((1,), dtype, scope="local")
                    acc[0] = T.cast(0.0, dtype)

                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                in_y = y + kh - pad
                                in_x = x + kw - pad
                                if 0 <= in_y < height and 0 <= in_x < width:
                                    acc[0] += Input[b, ic, in_y, in_x] * Weight[n, ic, kh, kw]

                    acc[0] = T.min(acc[0], T.cast(constant_value, dtype))
                    acc[0] = acc[0] + Bias[n, 0, 0]
                    acc[0] = acc[0] * T.cast(scaling_factor, dtype)
                    Output[b, n, y, x] = acc[0]

    return tilelang.compile(fused_conv_min_add_scale, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int):
        key = (batch_size, in_channels, out_channels, height, width, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_min_add_scale_kernel(
                batch_size, in_channels, out_channels, height, width, kernel_size,
                self.constant_value, self.scaling_factor
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        B, C, H, W = x.shape
        kernel = self._get_kernel(B, C, self.conv.out_channels, H, W, self.conv.kernel_size[0])
        weight = self.conv.weight.half()
        bias = self.bias.half()
        out = kernel(x.half(), weight, bias)
        return out