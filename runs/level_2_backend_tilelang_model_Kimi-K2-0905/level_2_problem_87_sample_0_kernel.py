import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_sub_mish_kernel(
    batch: int, out_channels: int, out_height: int, out_width: int,
    block_N: int = 64, block_H: int = 8, block_W: int = 8, threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def kernel(
        X: T.Tensor((batch, out_channels, out_height, out_width), dtype),
        C: T.Tensor((batch, out_channels, out_height, out_width), dtype),
        sub1: T.float32,
        sub2: T.float32,
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_W),
            T.ceildiv(out_height, block_H),
            T.ceildiv(out_channels, block_N),
            batch,
            threads=threads
        ) as (bx, by, bz, bb):
            start_w = bx * block_W
            start_h = by * block_H
            start_c = bz * block_N

            for local_c, local_h, local_w in T.Parallel(block_N, block_H, block_W):
                c = start_c + local_c
                h = start_h + local_h
                w = start_w + local_w

                if c < out_channels and h < out_height and w < out_width:
                    val = X[bb, c, h, w]
                    val = val - sub1
                    val = val - sub2
                    # Mish: x * tanh(softplus(x)) where softplus(x) = ln(1 + exp(x))
                    sp = T.log(1.0 + T.exp(val))
                    tanh_sp = T.tanh(sp)
                    C[bb, c, h, w] = val * tanh_sp

    return tilelang.compile(kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self._kernel_cache = {}

    def _get_kernel(self, b, c, h, w, tl_dtype: str):
        key = (b, c, h, w, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_sub_mish_kernel(b, c, h, w, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run convolution in FP16
        x = self.conv(x.half()).half()
        b, c, h, w = x.shape
        kernel = self._get_kernel(b, c, h, w, "float16")
        out = kernel(x, self.subtract_value_1, self.subtract_value_2)
        return out