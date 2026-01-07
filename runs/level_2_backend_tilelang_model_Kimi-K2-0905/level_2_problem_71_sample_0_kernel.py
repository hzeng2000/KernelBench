import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_div_leakyrelu_kernel(
    batch: int, in_channels: int, out_channels: int, height: int, width: int,
    kernel_size: int, stride: int = 1, padding: int = 0, divisor: float = 2.0,
    negative_slope: float = 0.01, block_M: int = 8, block_N: int = 16, block_K: int = 8,
    threads: int = 256, dtype: str = "float16"
):
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    @T.prim_func
    def kernel(
        Input: T.Tensor((batch, in_channels, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_N), T.ceildiv(out_height, block_M), batch * out_channels, threads=threads) as (bx, by, bz):
            tile_w = bx * block_N
            tile_h = by * block_M
            n = bz // batch
            b = bz % batch

            for local_h, local_w in T.Parallel(block_M, block_N):
                h = tile_h + local_h
                w = tile_w + local_w
                if h < out_height and w < out_width:
                    acc = 0.0
                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                ih = h * stride - padding + kh
                                iw = w * stride - padding + kw
                                if 0 <= ih < height and 0 <= iw < width:
                                    acc += Input[b, ic, ih, iw].astype("float32") * Weight[n, ic, kh, kw].astype("float32")
                    acc += Bias[n].astype("float32")
                    acc = acc / divisor
                    acc = T.max(acc, acc * negative_slope)
                    Output[b, n, h, w] = acc.astype(dtype)

    return tilelang.compile(kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self._kernel_cache = {}

    def _get_kernel(self, batch, in_channels, out_channels, height, width, kernel_size, stride, padding, divisor, dtype):
        key = (batch, in_channels, out_channels, height, width, kernel_size, stride, padding, divisor, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_div_leakyrelu_kernel(
                batch, in_channels, out_channels, height, width, kernel_size,
                stride, padding, divisor, negative_slope=0.01, dtype=dtype
            )
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.contiguous()
        batch, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding = self.conv.padding[0]

        weight = self.conv.weight.half()
        bias = self.conv.bias.half() if self.conv.bias is not None else torch.zeros(out_channels, dtype=torch.float16, device=x.device)

        kernel = self._get_kernel(batch, in_channels, height, width, kernel_size, stride, padding, self.divisor, "float16")
        output = kernel(x.half(), weight, bias)
        return output