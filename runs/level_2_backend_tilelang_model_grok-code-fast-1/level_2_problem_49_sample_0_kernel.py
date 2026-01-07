import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_softmax_sigmoid_kernel(B: int, C: int, D: int, H: int, W: int, dtype: str = "float16"):
    @T.prim_func
    def fused_softmax_sigmoid_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        Out: T.Tensor((B, C, D, H, W), dtype),
    ):
        with T.Kernel(B, D, H, W, threads=128) as b, dd, hh, ww:
            max_val = T.alloc_buffer((1,), dtype, scope="local")
            sum_exp = T.alloc_buffer((1,), dtype, scope="local")
            max_val[0] = T.reduce(T.max, A[b, T.reduce_axis(0, C), dd, hh, ww], axis=0, init=T.float16(-65504.0))
            sum_exp[0] = T.reduce(T.add, T.exp(A[b, T.reduce_axis(0, C), dd, hh, ww] - max_val[0]), axis=0, init=T.float16(0.0))
            for c in T.serial(0, C):
                exp_val = T.exp(A[b, c, dd, hh, ww] - max_val[0])
                softmax_val = exp_val / sum_exp[0]
                Out[b, c, dd, hh, ww] = T.float16(1.0) / (T.float16(1.0) + T.exp(-softmax_val))

    return tilelang.compile(fused_softmax_sigmoid_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies fused Softmax and Sigmoid using TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        # Precompute the kernel since shapes are fixed
        self.kernel = build_fused_softmax_sigmoid_kernel(16, 64, 32, 64, 64, dtype="float16")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x).half()
        x = self.kernel(x)
        return x