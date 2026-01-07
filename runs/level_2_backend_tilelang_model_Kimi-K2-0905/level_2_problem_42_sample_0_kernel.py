import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_conv_transpose_gap_logsumexp_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    in_height: int,
    in_width: int,
    kernel_size: int,
    block_size: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    out_height = (in_height - 1) * 1 - 2 * 0 + kernel_size
    out_width = (in_width - 1) * 1 - 2 * 0 + kernel_size

    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_channels, in_height, in_width), dtype),
        W: T.Tensor((in_channels, out_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels, 1, 1), dtype),
        Out: T.Tensor((batch_size, 1), dtype),
    ):
        with T.Kernel(batch_size, threads=threads) as b:
            max_val = T.alloc_fragment((1,), dtype)
            sum_exp = T.alloc_fragment((1,), dtype)
            max_val[0] = T.min_value(dtype)
            sum_exp[0] = T.cast(0.0, dtype)

            for oc in T.Parallel(out_channels):
                avg_val = T.alloc_fragment((1,), dtype)
                avg_val[0] = T.cast(0.0, dtype)

                for ic in range(in_channels):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            for oh in range(out_height):
                                for ow in range(out_width):
                                    ih = oh - kh
                                    iw = ow - kw
                                    if ih >= 0 and ih < in_height and iw >= 0 and iw < in_width:
                                        val = X[b, ic, ih, iw] * W[ic, oc, kh, kw]
                                        avg_val[0] = avg_val[0] + val

                avg_val[0] = avg_val[0] / (out_height * out_width)
                avg_val[0] = avg_val[0] + Bias[oc, 0, 0]

                if avg_val[0] > max_val[0]:
                    max_val[0] = avg_val[0]

            for oc in T.Parallel(out_channels):
                avg_val = T.alloc_fragment((1,), dtype)
                avg_val[0] = T.cast(0.0, dtype)

                for ic in range(in_channels):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            for oh in range(out_height):
                                for ow in range(out_width):
                                    ih = oh - kh
                                    iw = ow - kw
                                    if ih >= 0 and ih < in_height and iw >= 0 and iw < in_width:
                                        val = X[b, ic, ih, iw] * W[ic, oc, kh, kw]
                                        avg_val[0] = avg_val[0] + val

                avg_val[0] = avg_val[0] / (out_height * out_width)
                avg_val[0] = avg_val[0] + Bias[oc, 0, 0]

                exp_val = T.exp(avg_val[0] - max_val[0])
                sum_exp[0] = sum_exp[0] + exp_val

            Out[b, 0] = T.log(sum_exp[0]) + max_val[0]
            Out[b, 0] = Out[b, 0] * T.cast(10.0, dtype)

    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, in_height: int, in_width: int, kernel_size: int):
        key = (batch_size, in_channels, out_channels, in_height, in_width, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_transpose_gap_logsumexp_kernel(
                batch_size, in_channels, out_channels, in_height, in_width, kernel_size
            )
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.contiguous()
        batch_size, in_channels, in_height, in_width = x.shape
        kernel = self._get_kernel(batch_size, in_channels, self.conv_transpose.out_channels, in_height, in_width, self.conv_transpose.kernel_size[0])
        
        weight = self.conv_transpose.weight.transpose(0, 1).contiguous().half()
        bias = self.bias.contiguous().half()
        out = kernel(x.half(), weight, bias)
        return out