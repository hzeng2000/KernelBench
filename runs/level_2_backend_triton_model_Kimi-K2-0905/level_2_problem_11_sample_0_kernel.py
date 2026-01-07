import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_bn_tanh_max_gn_kernel(
    x_ptr, out_ptr,
    weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    gn_weight_ptr, gn_bias_ptr,
    batch_size, out_channels, out_h, out_w,
    stride_h, stride_w, pad_h, pad_w,
    kernel_h, kernel_w,
    groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = batch_size * out_channels * out_h * out_w
    for i in range(pid, n_elements, tl.num_programs(0)):
        n = i // (out_channels * out_h * out_w)
        rem = i % (out_channels * out_h * out_w)
        c = rem // (out_h * out_w)
        hw = rem % (out_h * out_w)
        h_out = hw // out_w
        w_out = hw % out_w

        # Transposed convolution compute
        acc = 0.0
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                h_in = h_out - kh + pad_h
                w_in = w_out - kw + pad_w
                if h_in >= 0 and w_in >= 0 and h_in < 32 and w_in < 32:
                    in_idx = n * (64 * 32 * 32) + c * (32 * 32) + h_in * 32 + w_in
                    acc += tl.load(x_ptr + in_idx)
        out_idx = i
        val = acc

        # BatchNorm
        mean = tl.load(running_mean_ptr + c)
        var = tl.load(running_var_ptr + c)
        bn_val = (val - mean) / tl.sqrt(var + eps)

        # Tanh
        tanh_val = tl.tanh(bn_val)

        # MaxPool (2x2, stride 2)
        h_pool = h_out // 2
        w_pool = w_out // 2
        pool_idx = n * (out_channels * 16 * 16) + c * (16 * 16) + h_pool * 16 + w_pool
        if h_out % 2 == 0 and w_out % 2 == 0:
            tl.store(out_ptr + pool_idx, tanh_val)

        # GroupNorm
        group_size = out_channels // groups
        group = c // group_size
        gn_idx = n * (groups * 16 * 16) + group * (16 * 16) + h_pool * 16 + w_pool
        gn_weight = tl.load(gn_weight_ptr + c)
        gn_bias = tl.load(gn_bias_ptr + c)
        gn_val = tanh_val * gn_weight + gn_bias
        if h_out % 2 == 0 and w_out % 2 == 0:
            tl.store(out_ptr + pool_idx, gn_val)


def fused_transpose_bn_tanh_max_gn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var,
                                   gn_weight, gn_bias, groups, eps):
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels = 128
    out_h = (in_h - 1) * 1 - 2 * 1 + 5
    out_w = (in_w - 1) * 1 - 2 * 1 + 5
    out_h_pool = out_h // 2
    out_w_pool = out_w // 2

    out = torch.empty(batch_size, out_channels, out_h_pool, out_w_pool, device=x.device, dtype=x.dtype)

    n_elements = batch_size * out_channels * out_h * out_w
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    fused_transpose_bn_tanh_max_gn_kernel[grid](
        x, out,
        conv_weight, conv_bias, bn_running_mean, bn_running_var,
        gn_weight, gn_bias,
        batch_size, out_channels, out_h, out_w,
        1, 1, 1, 1,
        5, 5,
        groups,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        return fused_transpose_bn_tanh_max_gn(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.batch_norm.eps,
        )