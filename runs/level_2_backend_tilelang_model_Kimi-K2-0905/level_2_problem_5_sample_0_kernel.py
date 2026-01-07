import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose_tanh_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    block_M: int = 8,
    block_N: int = 16,
    block_K: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    kH = kW = kernel_size
    sH = sW = stride
    pH = pW = padding
    opH = opW = output_padding

    @T.prim_func
    def kernel(
        Input: T.Tensor((batch_size, in_channels, in_height, in_width), dtype),
        Weight: T.Tensor((in_channels, out_channels, kH, kW), dtype),
        Bias: T.Tensor((out_channels, 1, 1), dtype),
        Output: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_N),
            T.ceildiv(out_height, block_M),
            batch_size * out_channels,
            threads=threads
        ) as (bx, by, bz):
            tile_w = bx * block_N
            tile_h = by * block_M
            n = bz // out_channels
            k = bz % out_channels

            bias_val = Bias[k, 0, 0]

            for dh, dw in T.Parallel(block_M, block_N):
                oh = tile_h + dh
                ow = tile_w + dw
                if oh < out_height and ow < out_width:
                    acc = 0.0
                    for c in range(in_channels):
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh + kh - kH + 1 - pH
                                iw = ow + kw - kW + 1 - pW
                                if (ih % sH == 0) and (iw % sW == 0):
                                    ih = ih // sH
                                    iw = iw // sW
                                    if 0 <= ih < in_height and 0 <= iw < in_width:
                                        acc += Input[n, c, ih, iw] * Weight[c, k, kh, kw]
                    acc -= bias_val
                    # tanh approximation
                    x2 = acc * acc
                    x4 = x2 * x2
                    x6 = x4 * x2
                    # tanh(x) ≈ x * (1 - x2/3 + x4*2/15 - x6*17/315)
                    tanh_approx = acc * (1.0 - x2 * (1.0 / 3.0) + x4 * (2.0 / 15.0) - x6 * (17.0 / 315.0))
                    Output[n, k, oh, ow] = tanh_approx

    return tilelang.compile(kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, in_height: int, in_width: int):
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        key = (batch_size, in_channels, out_channels, in_height, in_width, out_height, out_width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose_tanh_kernel(
                batch_size, in_channels, out_channels,
                in_height, in_width, out_height, out_width,
                self.kernel_size, self.stride, self.padding, self.output_padding,
                dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels = self.conv_transpose.out_channels

        weight = self.conv_transpose.weight.data.permute(1, 0, 2, 3).contiguous()
        bias = self.bias.contiguous()

        kernel = self._get_kernel(batch_size, in_channels, out_channels, in_height, in_width)
        output = kernel(x.half(), weight.half(), bias.half())
        return output.float()