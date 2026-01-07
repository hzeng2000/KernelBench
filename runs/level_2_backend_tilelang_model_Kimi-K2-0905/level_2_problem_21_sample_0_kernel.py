import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv_bias_scale_sigmoid_kernel(
    batch_size: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int = 3,
    in_channels: int = 8,
    block_M: int = 8,
    block_N: int = 32,
    block_K: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_height = height
    out_width = width

    @T.prim_func
    def fused_conv_bias_scale_sigmoid(
        Input: T.Tensor((batch_size, in_channels, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels, 1, 1), dtype),
        Scale: T.Tensor((out_channels, 1, 1), dtype),
        Output: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_N), T.ceildiv(out_height, block_M), batch_size * out_channels, threads=threads) as (bx, by, bz):
            n = bz // out_channels
            oc = bz % out_channels
            h_start = by * block_M
            w_start = bx * block_N

            for local_h, local_w in T.Parallel(block_M, block_N):
                h = h_start + local_h
                w = w_start + local_w
                if h < out_height and w < out_width:
                    acc = T.alloc_fragment((1,), dtype, scope="local")
                    acc[0] = T.cast(0.0, dtype)
                    for ic in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                ih = h + kh - pad
                                iw = w + kw - pad
                                if ih >= 0 and ih < height and iw >= 0 and iw < width:
                                    acc[0] += Input[n, ic, ih, iw] * Weight[oc, ic, kh, kw]
                    acc[0] += Bias[oc, 0, 0]
                    acc[0] *= Scale[oc, 0, 0]
                    acc[0] = T.sigmoid(acc[0])
                    Output[n, oc, h, w] = acc[0]

    return tilelang.compile(fused_conv_bias_scale_sigmoid, out_idx=[4], target="cuda")


def build_group_norm_kernel(
    batch_size: int,
    out_channels: int,
    height: int,
    width: int,
    num_groups: int,
    block_M: int = 8,
    block_N: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    channels_per_group = out_channels // num_groups

    @T.prim_func
    def group_norm_kernel(
        Input: T.Tensor((batch_size, out_channels, height, width), dtype),
        Output: T.Tensor((batch_size, out_channels, height, width), dtype),
    ):
        with T.Kernel(T.ceildiv(width, block_N), T.ceildiv(height, block_M), batch_size * num_groups, threads=threads) as (bx, by, bz):
            n = bz // num_groups
            g = bz % num_groups
            h_start = by * block_M
            w_start = bx * block_N

            # Compute mean
            sum_val = T.alloc_fragment((1,), "float32", scope="local")
            sum_val[0] = T.cast(0.0, "float32")
            count = channels_per_group * height * width
            for c in range(g * channels_per_group, (g + 1) * channels_per_group):
                for h in range(height):
                    for w in range(width):
                        sum_val[0] += T.cast(Input[n, c, h, w], "float32")
            mean = sum_val[0] / T.cast(count, "float32")

            # Compute variance
            var_sum = T.alloc_fragment((1,), "float32", scope="local")
            var_sum[0] = T.cast(0.0, "float32")
            for c in range(g * channels_per_group, (g + 1) * channels_per_group):
                for h in range(height):
                    for w in range(width):
                        diff = T.cast(Input[n, c, h, w], "float32") - mean
                        var_sum[0] += diff * diff
            var = var_sum[0] / T.cast(count, "float32")
            std = T.sqrt(var + T.cast(1e-5, "float32"))

            # Normalize
            for local_h, local_w in T.Parallel(block_M, block_N):
                h = h_start + local_h
                w = w_start + local_w
                if h < height and w < width:
                    for c in range(g * channels_per_group, (g + 1) * channels_per_group):
                        val = T.cast(Input[n, c, h, w], "float32")
                        norm_val = (val - mean) / std
                        Output[n, c, h, w] = T.cast(norm_val, dtype)

    return tilelang.compile(group_norm_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.num_groups = num_groups
        self._kernel_cache1 = {}
        self._kernel_cache2 = {}

    def _get_fused_kernel(self, batch_size: int, out_channels: int, height: int, width: int, in_channels: int, kernel_size: int):
        key = (batch_size, out_channels, height, width, in_channels, kernel_size)
        if key not in self._kernel_cache1:
            self._kernel_cache1[key] = build_fused_conv_bias_scale_sigmoid_kernel(
                batch_size, out_channels, height, width, kernel_size, in_channels
            )
        return self._kernel_cache1[key]

    def _get_norm_kernel(self, batch_size: int, out_channels: int, height: int, width: int, num_groups: int):
        key = (batch_size, out_channels, height, width, num_groups)
        if key not in self._kernel_cache2:
            self._kernel_cache2[key] = build_group_norm_kernel(
                batch_size, out_channels, height, width, num_groups
            )
        return self._kernel_cache2[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().half()
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # Fused conv + bias + scale + sigmoid
        kernel1 = self._get_fused_kernel(batch_size, out_channels, height, width, in_channels, kernel_size)
        weight = self.conv.weight.half()
        bias = self.bias.half()
        scale = self.scale.half()
        x = kernel1(x, weight, bias, scale)

        # Group norm
        kernel2 = self._get_norm_kernel(batch_size, out_channels, height, width, self.num_groups)
        x = kernel2(x)

        return x.float()