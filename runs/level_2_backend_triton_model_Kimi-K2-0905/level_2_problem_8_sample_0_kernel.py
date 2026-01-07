import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    depth, height, width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_depth, out_height, out_width,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)
    pid_od = tl.program_id(4)

    oc_start = pid_oc * BLOCK_OC
    oh_start = pid_oh * BLOCK_H
    ow_start = pid_ow * BLOCK_W
    od_start = pid_od * BLOCK_D

    acc = tl.zeros((BLOCK_OC, BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

    for ic_start in range(0, in_channels, BLOCK_IC):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    ic_offs = tl.arange(0, BLOCK_IC)
                    oc_offs = tl.arange(0, BLOCK_OC)
                    od_offs = od_start + tl.arange(0, BLOCK_D)
                    oh_offs = oh_start + tl.arange(0, BLOCK_H)
                    ow_offs = ow_start + tl.arange(0, BLOCK_W)

                    in_d = od_offs * stride_d - pad_d + kd
                    in_h = oh_offs * stride_h - pad_h + kh
                    in_w = ow_offs * stride_w - pad_w + kw

                    mask_d = (in_d >= 0) & (in_d < depth)
                    mask_h = (in_h >= 0) & (in_h < height)
                    mask_w = (in_w >= 0) & (in_w < width)

                    in_ptr = (
                        input_ptr
                        + pid_b * in_channels * depth * height * width
                        + (ic_start + ic_offs[:, None, None, None]) * depth * height * width
                        + in_d[None, :, None, None] * height * width
                        + in_h[None, None, :, None] * width
                        + in_w[None, None, None, :]
                    )
                    weight_ptr_off = (
                        weight_ptr
                        + (pid_oc * BLOCK_OC + oc_offs[:, None]) * in_channels * kernel_d * kernel_h * kernel_w
                        + (ic_start + ic_offs[None, :]) * kernel_d * kernel_h * kernel_w
                        + kd * kernel_h * kernel_w
                        + kh * kernel_w
                        + kw
                    )

                    in_vals = tl.load(in_ptr, mask=mask_d[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :], other=0.0)
                    w_vals = tl.load(weight_ptr_off)

                    for i in range(BLOCK_IC):
                        acc += w_vals[:, i, None, None, None] * in_vals[None, i, :, :, :]

    out_ptr = (
        output_ptr
        + pid_b * out_channels * out_depth * out_height * out_width
        + oc_start * out_depth * out_height * out_width
        + od_start * out_height * out_width
        + oh_start * out_width
        + ow_start
    )
    tl.store(out_ptr, acc)


@triton.jit
def fused_div_max_pool_gavg_kernel(
    x_ptr, out_ptr, divisor,
    batch_size, channels, in_d, in_h, in_w,
    pool_d, pool_h, pool_w,
    out_d, out_h, out_w,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    max_vals = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)

    for od in range(out_d):
        for oh in range(out_h):
            for ow in range(out_w):
                for pd in range(pool_d):
                    for ph in range(pool_h):
                        for pw in range(pool_w):
                            id = od * pool_d + pd
                            ih = oh * pool_h + ph
                            iw = ow * pool_w + pw
                            if id < in_d and ih < in_h and iw < in_w:
                                offs = (
                                    pid_b * channels * in_d * in_h * in_w
                                    + c_offs[:, None, None, None] * in_d * in_h * in_w
                                    + id * in_h * in_w
                                    + ih * in_w
                                    + iw
                                )
                                val = tl.load(x_ptr + offs)
                                max_vals = tl.maximum(max_vals, val)

    avg_val = tl.sum(max_vals) / (out_d * out_h * out_w)
    avg_val = avg_val / divisor

    out_offs = pid_b * channels + c_offs
    tl.store(out_ptr + out_offs, avg_val)


@triton.jit
def add_bias_sum_kernel(
    x_ptr, bias_ptr, out_ptr,
    batch_size, channels,
    sum_dim_size,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    offs = pid_b * channels + c_offs
    x_val = tl.load(x_ptr + offs)
    bias_val = tl.load(bias_ptr + c_offs)
    out_val = x_val + bias_val

    tl.store(out_ptr + offs, out_val)


def triton_conv3d(x, weight, bias=None, stride=1, padding=0):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    stride_d, stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride, stride)
    pad_d, pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding, padding)

    out_depth = (depth + 2 * pad_d - kernel_d) // stride_d + 1
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1

    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

    BLOCK_OC = 4
    BLOCK_IC = 4
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4

    grid = (
        batch_size,
        (out_channels + BLOCK_OC - 1) // BLOCK_OC,
        (out_height + BLOCK_H - 1) // BLOCK_H,
        (out_width + BLOCK_W - 1) // BLOCK_W,
        (out_depth + BLOCK_D - 1) // BLOCK_D,
    )

    conv3d_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_depth, out_height, out_width,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC
    )
    return out


def triton_fused_div_max_pool_gavg(x, divisor, pool_size):
    batch_size, channels, depth, height, width = x.shape
    pool_d, pool_h, pool_w = pool_size

    out_d = depth // pool_d
    out_h = height // pool_h
    out_w = width // pool_w

    out = torch.empty(batch_size, channels, device=x.device, dtype=x.dtype)

    BLOCK_C = 16
    grid = (batch_size, (channels + BLOCK_C - 1) // BLOCK_C)

    fused_div_max_pool_gavg_kernel[grid](
        x, out, divisor,
        batch_size, channels, depth, height, width,
        pool_d, pool_h, pool_w,
        out_d, out_h, out_w,
        BLOCK_C=BLOCK_C
    )
    return out


def triton_add_bias_sum(x, bias, sum_dim):
    batch_size, channels = x.shape
    sum_dim_size = 1

    out = torch.empty(batch_size, device=x.device, dtype=x.dtype)

    BLOCK_C = 16
    grid = (batch_size, (channels + BLOCK_C - 1) // BLOCK_C)

    add_bias_sum_kernel[grid](
        x, bias, out,
        batch_size, channels,
        sum_dim_size,
        BLOCK_C=BLOCK_C
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = triton_conv3d(x, self.conv_weight, stride=1, padding=0)
        x = triton_fused_div_max_pool_gavg(x, self.divisor, self.pool_size)
        x = x.view(x.size(0), -1, 1, 1, 1)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x