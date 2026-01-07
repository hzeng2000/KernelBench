import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    D_in: int,
    H_in: int,
    W_in: int,
    D_out: int,
    H_out: int,
    W_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    block_D: int = 4,
    block_H: int = 4,
    block_W: int = 4,
    block_OC: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def conv_transpose3d_kernel(
        Input: T.Tensor((batch_size, in_channels, D_in, H_in, W_in), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels, 1, 1, 1, 1), dtype),
        AddInput: T.Tensor((batch_size, out_channels, D_out, H_out, W_out), dtype),
        Output: T.Tensor((batch_size, out_channels, D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(
            T.ceildiv(W_out, block_W),
            T.ceildiv(H_out, block_H),
            T.ceildiv(D_out, block_D),
            T.ceildiv(out_channels, block_OC),
            batch_size,
            threads=threads
        ) as (bx, by, bz, boc, bb):
            start_w = bx * block_W
            start_h = by * block_H
            start_d = bz * block_D
            start_oc = boc * block_OC

            local_w = T.alloc_fragment((block_W,), "int32")
            local_h = T.alloc_fragment((block_H,), "int32")
            local_d = T.alloc_fragment((block_D,), "int32")
            local_oc = T.alloc_fragment((block_OC,), "int32")

            for i in T.Parallel(block_W):
                local_w[i] = start_w + i
            for i in T.Parallel(block_H):
                local_h[i] = start_h + i
            for i in T.Parallel(block_D):
                local_d[i] = start_d + i
            for i in T.Parallel(block_OC):
                local_oc[i] = start_oc + i

            acc = T.alloc_fragment((block_D, block_H, block_W, block_OC), dtype, scope="local")
            for d in T.Parallel(block_D):
                for h in T.Parallel(block_H):
                    for w in T.Parallel(block_W):
                        for oc in T.Parallel(block_OC):
                            acc[d, h, w, oc] = T.cast(0.0, dtype)

            for ic in range(in_channels):
                for kd in range(kernel_size):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            for d in T.Parallel(block_D):
                                for h in T.Parallel(block_H):
                                    for w in T.Parallel(block_W):
                                        for oc in T.Parallel(block_OC):
                                            in_d = (local_d[d] + padding - kd) // stride
                                            in_h = (local_h[h] + padding - kh) // stride
                                            in_w = (local_w[w] + padding - kw) // stride
                                            if (local_d[d] + padding - kd) % stride == 0 and \
                                               (local_h[h] + padding - kh) % stride == 0 and \
                                               (local_w[w] + padding - kw) % stride == 0 and \
                                               in_d >= 0 and in_d < D_in and \
                                               in_h >= 0 and in_h < H_in and \
                                               in_w >= 0 and in_w < W_in:
                                                acc[d, h, w, oc] += Input[bb, ic, in_d, in_h, in_w] * Weight[ic, oc, kd, kh, kw]

            for d in T.Parallel(block_D):
                for h in T.Parallel(block_H):
                    for w in T.Parallel(block_W):
                        for oc in T.Parallel(block_OC):
                            if local_d[d] < D_out and local_h[h] < H_out and local_w[w] < W_out and local_oc[oc] < out_channels:
                                val = acc[d, h, w, oc] + Bias[local_oc[oc], 0, 0, 0, 0]
                                add_val = AddInput[bb, local_oc[oc], local_d[d], local_h[h], local_w[w]]
                                val = val + add_val
                                # HardSwish: x * relu6(x + 3) / 6
                                relu6_val = T.min(T.max(val + T.cast(3.0, dtype), T.cast(0.0, dtype)), T.cast(6.0, dtype))
                                Output[bb, local_oc[oc], local_d[d], local_h[h], local_w[w]] = val * relu6_val * T.cast(1.0/6.0, dtype)

    return tilelang.compile(conv_transpose3d_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_kernel(self, batch_size, in_channels, out_channels, D_in, H_in, W_in, D_out, H_out, W_out, kernel_size, stride, padding, output_padding):
        key = (batch_size, in_channels, out_channels, D_in, H_in, W_in, D_out, H_out, W_out, kernel_size, stride, padding, output_padding)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_kernel(
                batch_size, in_channels, out_channels, D_in, H_in, W_in, D_out, H_out, W_out,
                kernel_size, stride, padding, output_padding
            )
        return self._kernel_cache[key]

    def forward(self, x, add_input):
        x_c = x.contiguous()
        add_input_c = add_input.contiguous()
        
        batch_size = x_c.shape[0]
        in_channels = x_c.shape[1]
        D_in, H_in, W_in = x_c.shape[2], x_c.shape[3], x_c.shape[4]
        
        D_out = (D_in - 1) * 2 - 2 * 1 + 3 + 1
        H_out = (H_in - 1) * 2 - 2 * 1 + 3 + 1
        W_out = (W_in - 1) * 2 - 2 * 1 + 3 + 1
        
        kernel = self._get_kernel(
            batch_size, in_channels, self.conv_transpose.out_channels,
            D_in, H_in, W_in, D_out, H_out, W_out,
            self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0],
            self.conv_transpose.padding[0], self.conv_transpose.output_padding[0]
        )
        
        weight = self.conv_transpose.weight.transpose(0, 1).contiguous()
        bias = self.bias.contiguous()
        
        output = kernel(x_c, weight, bias, add_input_c)
        
        return output