import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv_transpose_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    in_height: int,
    in_width: int,
    kernel_size: int,
    stride: int,
    out_height: int,
    out_width: int,
    block_M: int = 16,
    block_N: int = 16,
    block_K: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def fused_conv_transpose_kernel(
        Input: T.Tensor((batch_size, in_channels, in_height, in_width), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, out_channels, out_height, out_width), dtype),
        add_value: T.float32,
        multiply_value: T.float32
    ):
        with T.Kernel(T.ceildiv(out_height, block_M), T.ceildiv(out_width, block_N), batch_size * out_channels, threads=threads) as (by, bx, bz):
            out_y = by * block_M
            out_x = bx * block_N
            batch = bz // out_channels
            out_c = bz % out_channels

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = out_y + local_y
                x = out_x + local_x

                if y < out_height and x < out_width:
                    acc = 0.0
                    for in_c in T.serial(in_channels):
                        for ky in T.serial(kernel_size):
                            for kx in T.serial(kernel_size):
                                in_y = (y + ky) // stride
                                in_x = (x + kx) // stride
                                if (y + ky) % stride == 0 and (x + kx) % stride == 0 and in_y < in_height and in_x < in_width:
                                    acc += Input[batch, in_c, in_y, in_x] * Weight[in_c, out_c, ky, kx]

                    acc += Bias[out_c]
                    acc += add_value
                    acc = T.min(acc, 0.0)
                    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    gelu_x = acc
                    pi = 3.141592653589793
                    sqrt_2_over_pi = T.sqrt(2.0 / pi)
                    tanh_arg = sqrt_2_over_pi * (gelu_x + 0.044715 * gelu_x * gelu_x * gelu_x)
                    # tanh approximation
                    tanh = tanh_arg / (1.0 + T.abs(tanh_arg))
                    gelu_out = 0.5 * gelu_x * (1.0 + tanh)
                    Output[batch, out_c, y, x] = gelu_out * multiply_value

    return tilelang.compile(fused_conv_transpose_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, in_height: int, in_width: int, kernel_size: int, stride: int):
        out_height = (in_height - 1) * stride + kernel_size
        out_width = (in_width - 1) * stride + kernel_size
        key = (batch_size, in_channels, out_channels, in_height, in_width, kernel_size, stride)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_transpose_kernel(
                batch_size, in_channels, out_channels, in_height, in_width, kernel_size, stride, out_height, out_width
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - 1) * self.conv_transpose.stride[0] + self.conv_transpose.kernel_size[0]
        out_width = (in_width - 1) * self.conv_transpose.stride[1] + self.conv_transpose.kernel_size[1]

        kernel = self._get_kernel(batch_size, in_channels, self.conv_transpose.out_channels, in_height, in_width, self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0])
        
        weight = self.conv_transpose.weight.transpose(0, 1).contiguous()
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros(self.conv_transpose.out_channels, device=x.device, dtype=torch.float16)
        
        output = kernel(x.half(), weight.half(), bias.half(), self.add_value, self.multiply_value)
        return output