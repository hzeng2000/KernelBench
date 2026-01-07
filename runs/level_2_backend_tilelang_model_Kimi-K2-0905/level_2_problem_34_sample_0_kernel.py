import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_gelu_ln_kernel(
    batch_size: int,
    out_channels: int,
    out_D: int,
    out_H: int,
    out_W: int,
    groups: int = 1,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, out_channels, out_D, out_H, out_W), dtype),
        Weight: T.Tensor((out_channels,), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Scale: T.Tensor((1,), dtype),
        Out: T.Tensor((batch_size, out_channels, out_D, out_H, out_W), dtype),
    ):
        with T.Kernel(T.ceildiv(out_W * out_H * out_D, threads), batch_size, out_channels, threads=threads) as (spatial_blk, b, c):
            spatial_idx = spatial_blk * threads + T.thread_binding(0, threads, "threadIdx.x")
            if spatial_idx < out_W * out_H * out_D:
                d = spatial_idx // (out_H * out_W)
                hw = spatial_idx % (out_H * out_W)
                h = hw // out_W
                w = hw % out_W

                x_val = X[b, c, d, h, w]
                mean = T.reduce_sum(x_val, axis=[]) / 1.0
                var = T.reduce_sum((x_val - mean) * (x_val - mean), axis=[]) / 1.0
                inv_std = T.rsqrt(var + 1e-5)
                x_norm = (x_val - mean) * inv_std
                x_scaled = x_norm * Weight[c] + Bias[c]

                # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                pi = 3.141592653589793
                sqrt_2_over_pi = T.sqrt(2.0 / pi)
                x_cubed = x_scaled * x_scaled * x_scaled
                tanh_arg = sqrt_2_over_pi * (x_scaled + 0.044715 * x_cubed)
                tanh_val = T.tanh(tanh_arg)
                gelu_val = 0.5 * x_scaled * (1.0 + tanh_val)

                Out[b, c, d, h, w] = gelu_val * Scale[0]

    return tilelang.compile(kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, out_channels: int, out_D: int, out_H: int, out_W: int):
        key = (batch_size, out_channels, out_D, out_H, out_W)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_gelu_ln_kernel(batch_size, out_channels, out_D, out_H, out_W)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        batch_size, out_channels, out_D, out_H, out_W = x.shape

        # Get weight and bias from LayerNorm
        weight = self.layer_norm.weight.contiguous().half()
        bias = self.layer_norm.bias.contiguous().half()
        scale = torch.tensor([self.scaling_factor], dtype=torch.float16, device=x.device)

        kernel = self._get_kernel(batch_size, out_channels, out_D, out_H, out_W)
        x = kernel(x.contiguous().half(), weight, bias, scale)
        return x