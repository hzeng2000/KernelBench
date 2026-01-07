import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_mish_mish_kernel(
    batch: int, in_channels: int, out_channels: int, height: int, width: int,
    kernel_size: int, stride: int = 1, padding: int = 1,
    block_H: int = 8, block_W: int = 8, block_C: int = 64,
    threads: int = 256, dtype: str = "float16"
):
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    @T.prim_func
    def conv_mish_mish_kernel(
        Input: T.Tensor((batch, in_channels, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Output: T.Tensor((batch, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_height, block_H), T.ceildiv(out_width, block_W), T.ceildiv(out_channels, block_C), batch, threads=threads) as (by, bx, bc, bb):
            start_y = by * block_H
            start_x = bx * block_W
            start_c = bc * block_C
            start_b = bb

            for local_y, local_x, local_c in T.Parallel(block_H, block_W, block_C):
                y = start_y + local_y
                x = start_x + local_x
                c = start_c + local_c

                if y < out_height and x < out_width and c < out_channels:
                    acc = T.cast(0.0, dtype)
                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                in_y = y * stride - padding + kh
                                in_x = x * stride - padding + kw
                                if 0 <= in_y < height and 0 <= in_x < width:
                                    acc += Input[start_b, ic, in_y, in_x] * Weight[c, ic, kh, kw]
                    # Mish activation: x * tanh(softplus(x)) where softplus(x) = ln(1 + exp(x))
                    # First Mish
                    sp1 = T.log(T.cast(1.0, dtype) + T.exp(acc))
                    tanh1 = T.tanh(sp1)
                    mish1 = acc * tanh1
                    # Second Mish
                    sp2 = T.log(T.cast(1.0, dtype) + T.exp(mish1))
                    tanh2 = T.tanh(sp2)
                    mish2 = mish1 * tanh2
                    Output[start_b, c, y, x] = mish2

    return tilelang.compile(conv_mish_mish_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int):
        key = (batch, in_channels, out_channels, height, width, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_mish_mish_kernel(batch, in_channels, out_channels, height, width, kernel_size)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight.half()
        bias = self.conv.bias
        batch, _, height, width = x.shape
        x = x.half().contiguous()
        kernel = self._get_kernel(batch, x.size(1), weight.size(0), height, width, weight.size(2))
        out = kernel(x, weight)
        return out