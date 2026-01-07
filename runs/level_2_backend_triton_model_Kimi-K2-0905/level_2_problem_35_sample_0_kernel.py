import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_sub_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, in_h, in_w,
    out_c, out_h, out_w,
    kernel_size, stride, padding,
    subtract_value,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    c_start = pid_c * BLOCK_SIZE_C
    hw_start = pid_hw * BLOCK_SIZE_HW

    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)

    mask_c = c_offsets < out_c
    mask_hw = hw_offsets < (out_h * out_w)

    h_offsets = hw_offsets // out_w
    w_offsets = hw_offsets % out_w

    acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_HW), dtype=tl.float32)

    for ic in range(in_c):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = h_offsets * stride - padding + kh
                iw = w_offsets * stride - padding + kw

                mask_ih = (ih >= 0) & (ih < in_h)
                mask_iw = (iw >= 0) & (iw < in_w)
                mask_in = mask_ih & mask_iw

                x_idx = pid_b * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw
                w_idx = c_offsets * in_c * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw

                x_val = tl.load(x_ptr + x_idx, mask=mask_in, other=0.0)
                w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)

                acc += x_val[None, :] * w_val[:, None]

    b_val = tl.load(b_ptr + c_offsets, mask=mask_c, other=0.0)
    acc = acc + b_val[:, None]

    acc = acc - subtract_value

    # HardSwish
    acc = acc * tl.min(tl.max(acc + 3.0, 0.0), 6.0) / 6.0

    out_idx = pid_b * out_c * out_h * out_w + c_offsets[:, None] * out_h * out_w + h_offsets[None, :] * out_w + w_offsets[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask_c[:, None] & mask_hw[None, :])


@triton.jit
def maxpool_mish_kernel(
    x_ptr, out_ptr,
    batch, c, in_h, in_w,
    out_h, out_w,
    kernel_size, stride,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    c_start = pid_c * BLOCK_SIZE_C
    hw_start = pid_hw * BLOCK_SIZE_HW

    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)

    mask_c = c_offsets < c
    mask_hw = hw_offsets < (out_h * out_w)

    oh = hw_offsets // out_w
    ow = hw_offsets % out_w

    acc = tl.full((BLOCK_SIZE_C, BLOCK_SIZE_HW), float('-inf'), dtype=tl.float32)

    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ih = oh * stride + kh
            iw = ow * stride + kw

            mask_ih = (ih < in_h)
            mask_iw = (iw < in_w)
            mask_in = mask_ih & mask_iw

            x_idx = pid_b * c * in_h * in_w + c_offsets[:, None] * in_h * in_w + ih[None, :] * in_w + iw[None, :]
            x_val = tl.load(x_ptr + x_idx, mask=mask_c[:, None] & mask_hw[None, :] & mask_in[None, :], other=float('-inf'))

            acc = tl.maximum(acc, x_val)

    # Mish activation
    acc = acc * tl.tanh(tl.nn.relu(acc + 4.0))

    out_idx = pid_b * c * out_h * out_w + c_offsets[:, None] * out_h * out_w + oh[None, :] * out_w + ow[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask_c[:, None] & mask_hw[None, :])


def triton_conv_sub_hardswish(x, weight, bias, subtract_value, stride=1, padding=0):
    batch, in_c, in_h, in_w = x.shape
    out_c, _, kernel_size, _ = weight.shape
    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    out = torch.empty(batch, out_c, out_h, out_w, device=x.device, dtype=x.dtype)

    BLOCK_SIZE_C = 16
    BLOCK_SIZE_HW = 32

    grid = (batch, (out_c + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C, (out_h * out_w + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW)

    fused_conv_sub_hardswish_kernel[grid](
        x, weight, bias, out,
        batch, in_c, in_h, in_w,
        out_c, out_h, out_w,
        kernel_size, stride, padding,
        subtract_value,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    return out


def triton_maxpool_mish(x, kernel_size, stride):
    batch, c, in_h, in_w = x.shape
    out_h = (in_h - kernel_size) // stride + 1
    out_w = (in_w - kernel_size) // stride + 1

    out = torch.empty(batch, c, out_h, out_w, device=x.device, dtype=x.dtype)

    BLOCK_SIZE_C = 16
    BLOCK_SIZE_HW = 32

    grid = (batch, (c + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C, (out_h * out_w + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW)

    maxpool_mish_kernel[grid](
        x, out,
        batch, c, in_h, in_w,
        out_h, out_w,
        kernel_size, stride,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = triton_conv_sub_hardswish(x, self.conv.weight, self.conv.bias, self.subtract_value, stride=1, padding=0)
        x = triton_maxpool_mish(x, self.pool_kernel_size, self.pool_kernel_size)
        return x