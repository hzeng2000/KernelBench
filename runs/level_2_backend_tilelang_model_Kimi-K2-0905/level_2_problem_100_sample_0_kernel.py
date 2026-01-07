import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth_out: int,
    height_out: int,
    width_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    min_value: float,
    divisor: float,
    block_d: int = 4,
    block_h: int = 8,
    block_w: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    kernel_extent = kernel_size
    stride_val = stride
    pad = padding
    
    @T.prim_func
    def conv_transpose3d_kernel(
        Input: T.Tensor((batch_size, in_channels, depth_out // stride_val, height_out // stride_val, width_out // stride_val), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_extent, kernel_extent, kernel_extent), dtype),
        Output: T.Tensor((batch_size, out_channels, depth_out, height_out, width_out), dtype),
    ):
        with T.Kernel(T.ceildiv(width_out, block_w), T.ceildiv(height_out, block_h), T.ceildiv(depth_out, block_d), batch_size, threads=threads) as (bx, by, bz, bbatch):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            batch_idx = bbatch

            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                out_d = start_d + local_d
                out_h = start_h + local_h
                out_w = start_w + local_w

                if out_d < depth_out and out_h < height_out and out_w < width_out:
                    acc = T.alloc_fragment((out_channels,), dtype, scope="local")
                    for oc in T.Parallel(out_channels):
                        acc[oc] = T.cast(0.0, dtype)

                    for ic in range(in_channels):
                        for kd in range(kernel_extent):
                            for kh in range(kernel_extent):
                                for kw in range(kernel_extent):
                                    in_d = (out_d + pad - kd) // stride_val
                                    in_h = (out_h + pad - kh) // stride_val
                                    in_w = (out_w + pad - kw) // stride_val

                                    if (out_d + pad - kd) % stride_val == 0 and \
                                       (out_h + pad - kh) % stride_val == 0 and \
                                       (out_w + pad - kw) % stride_val == 0 and \
                                       in_d >= 0 and in_d < depth_out // stride_val and \
                                       in_h >= 0 and in_h < height_out // stride_val and \
                                       in_w >= 0 and in_w < width_out // stride_val:
                                        for oc in T.Parallel(out_channels):
                                            acc[oc] += Input[batch_idx, ic, in_d, in_h, in_w] * Weight[ic, oc, kd, kh, kw]

                    for oc in T.Parallel(out_channels):
                        val = acc[oc]
                        val = T.max(val, T.cast(min_value, dtype))
                        val = val / T.cast(divisor, dtype)
                        Output[batch_idx, oc, out_d, out_h, out_w] = val

    return tilelang.compile(conv_transpose3d_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size, dtype=torch.float16))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, depth_out: int, height_out: int, width_out: int):
        key = (batch_size, depth_out, height_out, width_out)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_kernel(
                batch_size=batch_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                depth_out=depth_out,
                height_out=height_out,
                width_out=width_out,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                min_value=self.min_value,
                divisor=self.divisor,
                dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().half()
        batch_size, _, depth_in, height_in, width_in = x.shape

        depth_out = (depth_in - 1) * self.stride - 2 * self.padding + self.kernel_size
        height_out = (height_in - 1) * self.stride - 2 * self.padding + self.kernel_size
        width_out = (width_in - 1) * self.stride - 2 * self.padding + self.kernel_size

        kernel = self._get_kernel(batch_size, depth_out, height_out, width_out)
        output = kernel(x, self.weight)

        return output