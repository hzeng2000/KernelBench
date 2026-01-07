import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_size, stride, padding,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    oc_start = pid_oc * BLOCK_C
    dhw_start = pid_dhw * BLOCK_DHW

    oc_range = oc_start + tl.arange(0, BLOCK_C)
    dhw_range = dhw_start + tl.arange(0, BLOCK_DHW)

    mask_oc = oc_range < out_channels
    mask_dhw = dhw_range < (out_d * out_h * out_w)

    for b in range(batch_size):
        for idx_dhw in range(BLOCK_DHW):
            if dhw_start + idx_dhw >= out_d * out_h * out_w:
                continue
            out_d_idx = (dhw_start + idx_dhw) // (out_h * out_w)
            rem = (dhw_start + idx_dhw) % (out_h * out_w)
            out_h_idx = rem // out_w
            out_w_idx = rem % out_w

            in_d_start = out_d_idx * stride - padding
            in_h_start = out_h_idx * stride - padding
            in_w_start = out_w_idx * stride - padding

            acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

            for ic in range(0, in_channels, BLOCK_C):
                ic_range = ic + tl.arange(0, BLOCK_C)
                mask_ic = ic_range < in_channels

                for kd in range(kernel_size):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            in_d_idx = in_d_start + kd
                            in_h_idx = in_h_start + kh
                            in_w_idx = in_w_start + kw

                            if in_d_idx >= 0 and in_d_idx < in_d and in_h_idx >= 0 and in_h_idx < in_h and in_w_idx >= 0 and in_w_idx < in_w:
                                in_offset = b * in_channels * in_d * in_h * in_w + ic_range * in_d * in_h * in_w + in_d_idx * in_h * in_w + in_h_idx * in_w + in_w_idx
                                weight_offset = oc_range * in_channels * kernel_size * kernel_size * kernel_size + ic_range * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw

                                in_val = tl.load(input_ptr + in_offset, mask=mask_ic, other=0.0)
                                weight_val = tl.load(weight_ptr + weight_offset, mask=mask_oc[:, None] & mask_ic[None, :], other=0.0)
                                acc += tl.sum(in_val * weight_val, axis=1)

            out_offset = b * out_channels * out_d * out_h * out_w + oc_range * out_d * out_h * out_w + (dhw_start + idx_dhw)
            tl.store(output_ptr + out_offset, acc, mask=mask_oc)


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    batch_size, channels, d, h, w,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    d_idx = pid_dhw // (h * w)
    rem = pid_dhw % (h * w)
    h_idx = rem // w
    w_idx = rem % w

    max_val = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)

    for c in range(0, channels, BLOCK_C):
        c_range = c + tl.arange(0, BLOCK_C)
        mask_c = c_range < channels
        offset = pid_b * channels * d * h * w + c_range * d * h * w + d_idx * h * w + h_idx * w + w_idx
        val = tl.load(input_ptr + offset, mask=mask_c, other=float('-inf'))
        max_val = tl.maximum(max_val, val)

    max_val = tl.max(max_val)

    exp_sum = 0.0
    for c in range(0, channels, BLOCK_C):
        c_range = c + tl.arange(0, BLOCK_C)
        mask_c = c_range < channels
        offset = pid_b * channels * d * h * w + c_range * d * h * w + d_idx * h * w + h_idx * w + w_idx
        val = tl.load(input_ptr + offset, mask=mask_c, other=0.0)
        exp_val = tl.exp(val - max_val)
        exp_sum += tl.sum(exp_val, axis=0)

    for c in range(0, channels, BLOCK_C):
        c_range = c + tl.arange(0, BLOCK_C)
        mask_c = c_range < channels
        offset = pid_b * channels * d * h * w + c_range * d * h * w + d_idx * h * w + h_idx * w + w_idx
        val = tl.load(input_ptr + offset, mask=mask_c, other=0.0)
        out_val = tl.exp(val - max_val) / exp_sum
        tl.store(output_ptr + offset, out_val, mask=mask_c)


@triton.jit
def maxpool3d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_size, stride,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    c_start = pid_c * BLOCK_C
    dhw_start = pid_dhw * BLOCK_DHW

    oc_range = c_start + tl.arange(0, BLOCK_C)
    mask_oc = oc_range < channels

    for idx_dhw in range(BLOCK_DHW):
        if dhw_start + idx_dhw >= out_d * out_h * out_w:
            continue
        out_d_idx = (dhw_start + idx_dhw) // (out_h * out_w)
        rem = (dhw_start + idx_dhw) % (out_h * out_w)
        out_h_idx = rem // out_w
        out_w_idx = rem % out_w

        in_d_start = out_d_idx * stride
        in_h_start = out_h_idx * stride
        in_w_start = out_w_idx * stride

        max_val = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)

        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_d_idx = in_d_start + kd
                    in_h_idx = in_h_start + kh
                    in_w_idx = in_w_start + kw

                    if in_d_idx < in_d and in_h_idx < in_h and in_w_idx < in_w:
                        in_offset = pid_b * channels * in_d * in_h * in_w + oc_range * in_d * in_h * in_w + in_d_idx * in_h * in_w + in_h_idx * in_w + in_w_idx
                        val = tl.load(input_ptr + in_offset, mask=mask_oc, other=float('-inf'))
                        max_val = tl.maximum(max_val, val)

        out_offset = pid_b * channels * out_d * out_h * out_w + oc_range * out_d * out_h * out_w + (dhw_start + idx_dhw)
        tl.store(output_ptr + out_offset, max_val, mask=mask_oc)


def triton_conv3d(input, weight, bias=None, stride=1, padding=0):
    batch_size, in_channels, in_d, in_h, in_w = input.shape
    out_channels, _, kernel_size, _, _ = weight.shape
    out_d = (in_d + 2 * padding - kernel_size) // stride + 1
    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=input.device, dtype=input.dtype)

    BLOCK_C = 16
    BLOCK_DHW = 64

    grid = (batch_size, (out_channels + BLOCK_C - 1) // BLOCK_C, (out_d * out_h * out_w + BLOCK_DHW - 1) // BLOCK_DHW)

    conv3d_kernel[grid](
        input, weight, bias, output,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding,
        BLOCK_C=BLOCK_C, BLOCK_DHW=BLOCK_DHW
    )
    return output


def triton_softmax(input, dim):
    assert dim == 1
    batch_size, channels, d, h, w = input.shape
    output = torch.empty_like(input)

    BLOCK_C = 32

    grid = (batch_size, d * h * w)

    softmax_kernel[grid](
        input, output,
        batch_size, channels, d, h, w,
        BLOCK_C=BLOCK_C
    )
    return output


def triton_maxpool3d(input, kernel_size, stride=None):
    if stride is None:
        stride = kernel_size
    batch_size, channels, in_d, in_h, in_w = input.shape
    out_d = (in_d - kernel_size) // stride + 1
    out_h = (in_h - kernel_size) // stride + 1
    out_w = (in_w - kernel_size) // stride + 1

    output = torch.empty(batch_size, channels, out_d, out_h, out_w, device=input.device, dtype=input.dtype)

    BLOCK_C = 16
    BLOCK_DHW = 64

    grid = (batch_size, (channels + BLOCK_C - 1) // BLOCK_C, (out_d * out_h * out_w + BLOCK_DHW - 1) // BLOCK_DHW)

    maxpool3d_kernel[grid](
        input, output,
        batch_size, channels, in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride,
        BLOCK_C=BLOCK_C, BLOCK_DHW=BLOCK_DHW
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = triton_conv3d(x, self.weight, self.bias, stride=1, padding=0)
        x = triton_softmax(x, dim=1)
        x = triton_maxpool3d(x, self.pool_kernel_size)
        x = triton_maxpool3d(x, self.pool_kernel_size)
        return x