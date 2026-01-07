import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_relu_bias_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    block_size: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_height = height
    out_width = width

    @T.prim_func
    def conv_relu_bias_kernel(
        Input: T.Tensor((batch_size, in_channels, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels, 1, 1), dtype),
        Output: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_size), T.ceildiv(out_height, block_size), batch_size * out_channels, threads=threads) as (bx, by, bz):
            b = bz // out_channels
            oc = bz % out_channels

            start_x = bx * block_size
            start_y = by * block_size

            for local_y, local_x in T.Parallel(block_size, block_size):
                y = start_y + local_y
                x = start_x + local_x

                if y < out_height and x < out_width:
                    acc = T.cast(0.0, dtype)
                    for ic in T.Range(in_channels):
                        for kh in T.Range(kernel_size):
                            for kw in T.Range(kernel_size):
                                ih = y + kh - pad
                                iw = x + kw - pad
                                if 0 <= ih < height and 0 <= iw < width:
                                    acc += Input[b, ic, ih, iw] * Weight[oc, ic, kh, kw]
                    acc += Bias[oc, 0, 0]
                    Output[b, oc, y, x] = T.max(acc, T.cast(0.0, dtype))

    return tilelang.compile(conv_relu_bias_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def _get_kernel(self, batch_size: int, height: int, width: int):
        key = (batch_size, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_relu_bias_kernel(
                batch_size,
                self.in_channels,
                self.out_channels,
                height,
                width,
                self.kernel_size,
                dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        batch_size, _, height, width = x.shape
        kernel = self._get_kernel(batch_size, height, width)
        weight = self.conv.weight.half()
        bias = self.bias.half()
        out = kernel(x, weight, bias)
        return out.float()