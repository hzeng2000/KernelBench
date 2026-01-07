import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_OC: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    h_start = pid_h * stride - padding
    w_start = pid_w * stride - padding

    acc = 0.0
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                h_in = h_start + kh
                w_in = w_start + kw
                if h_in >= 0 and h_in < height and w_in >= 0 and w_in < width:
                    x_idx = pid_b * in_channels * height * width + ic * height * width + h_in * width + w_in
                    w_idx = pid_oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    if b_ptr:
        b_val = tl.load(b_ptr + pid_oc)
        acc += b_val

    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1
    out_idx = pid_b * out_channels * out_h * out_w + pid_oc * out_h * out_w + pid_h * out_w + pid_w
    tl.store(out_ptr + out_idx, acc)


@triton.jit
def instance_norm_kernel(
    x_ptr, out_ptr, mean_ptr, var_ptr,
    batch_size, channels, height, width,
    eps, divide_by,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Compute mean
    sum_val = 0.0
    numel = height * width
    for hw in range(0, numel, BLOCK_SIZE):
        offsets = hw + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        h = offsets // width
        w = offsets % width
        idx = pid_b * channels * height * width + pid_c * height * width + h * width + w
        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x_val)

    mean_val = sum_val / numel
    tl.store(mean_ptr + pid_b * channels + pid_c, mean_val)

    # Compute variance
    sum_sq = 0.0
    for hw in range(0, numel, BLOCK_SIZE):
        offsets = hw + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        h = offsets // width
        w = offsets % width
        idx = pid_b * channels * height * width + pid_c * height * width + h * width + w
        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_sq += tl.sum((x_val - mean_val) * (x_val - mean_val))

    var_val = sum_sq / numel
    tl.store(var_ptr + pid_b * channels + pid_c, var_val)

    # Normalize and divide
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    for hw in range(0, numel, BLOCK_SIZE):
        offsets = hw + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        h = offsets // width
        w = offsets % width
        idx = pid_b * channels * height * width + pid_c * height * width + h * width + w
        x_val = tl.load(x_ptr + idx, mask=mask)
        out_val = (x_val - mean_val) * inv_std / divide_by
        tl.store(out_ptr + idx, out_val, mask=mask)


def triton_conv2d(x, weight, bias, stride, padding):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)

    grid = (batch_size, out_channels, out_h, out_w)
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 1
    BLOCK_SIZE_OC = 1

    conv_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_OC=BLOCK_SIZE_OC
    )
    return out


def triton_instance_norm_and_divide(x, divide_by):
    batch_size, channels, height, width = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    mean = torch.empty(batch_size, channels, dtype=x.dtype, device=x.device)
    var = torch.empty(batch_size, channels, dtype=x.dtype, device=x.device)

    grid = (batch_size, channels)
    BLOCK_SIZE = 128
    eps = 1e-5

    instance_norm_kernel[grid](
        x, out, mean, var,
        batch_size, channels, height, width,
        eps, divide_by,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by

    def forward(self, x):
        x = triton_conv2d(x, self.conv.weight, self.conv.bias, stride=1, padding=1)
        x = triton_instance_norm_and_divide(x, self.divide_by)
        return x