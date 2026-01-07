import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_min_kernel(B: int, C: int, H: int, W: int, dtype: str = "float16"):
    @T.prim_func
    def min_kernel(
        A: T.Tensor((B, C, H, W), dtype),
        C_out: T.Tensor((B, 1, H, W), dtype),
    ):
        for b, h, w in T.grid(B, H, W):
            C_out[b, 0, h, w] = T.reduce(T.min, A[b, T.reduce_axis(0, C), h, w], init=T.max_value(dtype))
    return tilelang.compile(min_kernel, out_idx=[1], target="cuda")


def build_sum_kernel(B: int, H: int, W: int, dtype: str = "float16"):
    @T.prim_func
    def sum_kernel(
        A: T.Tensor((B, 1, H, W), dtype),
        C: T.Tensor((B, 1, 1, W), dtype),
    ):
        for b, w in T.grid(B, W):
            C[b, 0, 0, w] = T.reduce(T.add, A[b, 0, T.reduce_axis(0, H), w], init=0.0)
    return tilelang.compile(sum_kernel, out_idx=[1], target="cuda")


def build_gelu_add_kernel(B: int, W: int, dtype: str = "float16"):
    @T.prim_func
    def gelu_add_kernel(
        A: T.Tensor((B, 1, 1, W), dtype),
        bias: T.Tensor((1, 1, 1, 1), dtype),
        C: T.Tensor((B, 1, 1, W), dtype),
    ):
        for b, w in T.grid(B, W):
            x = A[b, 0, 0, w]
            gelu_x = 0.5 * x * (1.0 + T.erf(x / T.sqrt(2.0)))
            C[b, 0, 0, w] = gelu_x + bias[0, 0, 0, 0]
    return tilelang.compile(gelu_add_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Assuming fixed shapes from the problem
        B, C, H_in, W_in = 16, 64, 128, 128
        H_out, W_out = 256, 256  # After conv_transpose
        self.min_kernel = build_min_kernel(B, out_channels, H_out, W_out, dtype="float16")
        self.sum_kernel = build_sum_kernel(B, H_out, W_out, dtype="float16")
        self.gelu_add_kernel = build_gelu_add_kernel(B, W_out, dtype="float16")

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.min_kernel(x)
        x = self.sum_kernel(x)
        x = self.gelu_add_kernel(x, self.bias)
        return x