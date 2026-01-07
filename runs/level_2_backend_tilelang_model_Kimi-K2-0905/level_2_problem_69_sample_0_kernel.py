import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_hardswish_relu_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    block_h: int = 8,
    block_w: int = 8,
    block_out: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_height = height
    out_width = width

    @T.prim_func
    def conv_hardswish_relu_kernel(
        X: T.Tensor((batch_size, in_channels, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        Y: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_w), T.ceildiv(out_height, block_h), T.ceildiv(batch_size * out_channels, block_out), threads=threads) as (bx, by, bz):
            start_x = bx * block_w
            start_y = by * block_h
            start_z = bz * block_out

            for local_z, local_y, local_x in T.Parallel(block_out, block_h, block_w):
                z = start_z + local_z
                y = start_y + local_y
                x = start_x + local_x

                if z < batch_size * out_channels and y < out_height and x < out_width:
                    out_c = z % out_channels
                    n = z // out_channels

                    acc = 0.0
                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                h_in = y + kh - pad
                                w_in = x + kw - pad
                                if 0 <= h_in < height and 0 <= w_in < width:
                                    acc += X[n, ic, h_in, w_in] * W[out_c, ic, kh, kw]
                    acc += B[out_c]

                    # HardSwish
                    relu6 = T.min(T.max(acc + 3.0, 0.0), 6.0)
                    hardswish = acc * relu6 / 6.0

                    # ReLU
                    Y[n, out_c, y, x] = T.max(hardswish, 0.0)

    return tilelang.compile(conv_hardswish_relu_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int):
        key = (batch_size, in_channels, out_channels, height, width, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_hardswish_relu_kernel(batch_size, in_channels, out_channels, height, width, kernel_size)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and in fp16
        x = x.contiguous().half()
        batch_size, in_channels, height, width = x.shape

        # Get weight and bias from conv layer
        weight = self.conv.weight.half()
        bias = self.conv.bias.half() if self.conv.bias is not None else torch.zeros(self.conv.out_channels, dtype=torch.float16, device=x.device)

        kernel = self._get_kernel(batch_size, in_channels, self.conv.out_channels, height, width, self.conv.kernel_size[0])
        output = kernel(x, weight, bias)

        return output