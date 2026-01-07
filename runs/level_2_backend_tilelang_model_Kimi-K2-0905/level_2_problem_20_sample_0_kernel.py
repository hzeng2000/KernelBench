import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_transpose_conv_kernel(
    batch: int, in_c: int, out_c: int, in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int, k_d: int, k_h: int, k_w: int,
    stride_d: int, stride_h: int, stride_w: int, pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int, threads: int = 256, dtype: str = "float16"
):
    @T.prim_func
    def fused_transpose_conv_kernel(
        Input: T.Tensor((batch, in_c, in_d, in_h, in_w), dtype),
        Weight: T.Tensor((in_c, out_c, k_d, k_h, k_w), dtype),
        Bias: T.Tensor((out_c, 1, 1, 1), dtype),
        Output: T.Tensor((batch, out_c, out_d, out_h, out_w), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w, 8), T.ceildiv(out_h, 8), T.ceildiv(out_d, 8), batch, threads=threads) as (bx, by, bz, b_n):
            for local_w, local_h, local_d in T.Parallel(8, 8, 8):
                out_w_idx = bx * 8 + local_w
                out_h_idx = by * 8 + local_h
                out_d_idx = bz * 8 + local_d
                if out_w_idx < out_w and out_h_idx < out_h and out_d_idx < out_d:
                    for oc in T.Parallel(out_c):
                        acc = T.cast(0.0, dtype)
                        for ic in T.Range(in_c):
                            for kd in T.Range(k_d):
                                for kh in T.Range(k_h):
                                    for kw in T.Range(k_w):
                                        in_d_idx = (out_d_idx - kd + k_d - 1 - pad_d - out_pad_d) // stride_d
                                        in_h_idx = (out_h_idx - kh + k_h - 1 - pad_h - out_pad_h) // stride_h
                                        in_w_idx = (out_w_idx - kw + k_w - 1 - pad_w - out_pad_w) // stride_w
                                        if (in_d_idx >= 0 and in_d_idx < in_d and
                                            in_h_idx >= 0 and in_h_idx < in_h and
                                            in_w_idx >= 0 and in_w_idx < in_w):
                                            acc += Input[b_n, ic, in_d_idx, in_h_idx, in_w_idx] * Weight[ic, oc, kd, kh, kw]
                        val = acc + Bias[oc, 0, 0, 0]
                        original = val
                        val = val + original
                        val = val * original
                        val = val + original
                        Output[b_n, oc, out_d_idx, out_h_idx, out_w_idx] = val
    return tilelang.compile(fused_transpose_conv_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.bias_shape = bias_shape

        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_kernel(self, batch, in_d, in_h, in_w, out_d, out_h, out_w, tl_dtype):
        key = (batch, in_d, in_h, in_w, out_d, out_h, out_w, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_transpose_conv_kernel(
                batch, self.in_channels, self.out_channels,
                in_d, in_h, in_w, out_d, out_h, out_w,
                *self.kernel_size, *self.stride, *self.padding, *self.output_padding,
                dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch, _, in_d, in_h, in_w = x.shape
        out_d = (in_d - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        out_h = (in_h - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        out_w = (in_w - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2] + self.output_padding[2]

        kernel = self._get_kernel(batch, in_d, in_h, in_w, out_d, out_h, out_w, "float16")
        output = kernel(x.half(), self.weight.half(), self.bias.half())
        return output