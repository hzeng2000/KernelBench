import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_h, in_w,
    out_channels, out_h, out_w,
    kernel_size, stride, padding, output_padding,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_outputs = batch_size * out_channels * out_h * out_w
    if pid * BLOCK_SIZE >= num_outputs:
        return

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_outputs

    # Compute output indices
    n = offsets // (out_channels * out_h * out_w)
    remainder = offsets % (out_channels * out_h * out_w)
    c_out = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    h_out = remainder // out_w
    w_out = remainder % out_w

    # Compute input region
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding

    acc = 0.0
    for c_in in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                h_in = h_in_start + kh
                w_in = w_in_start + kw
                if h_in >= 0 and h_in < in_h and w_in >= 0 and w_in < in_w:
                    input_idx = n * in_channels * in_h * in_w + c_in * in_h * in_w + h_in * in_w + w_in
                    weight_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += input_val * weight_val

    output_idx = n * out_channels * out_h * out_w + c_out * out_h * out_w + h_out * out_w + w_out
    tl.store(output_ptr + output_idx, acc, mask=mask)


@triton.jit
def multiply_and_double_mean_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    multiplier,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_elements = batch_size * channels
    if pid * BLOCK_SIZE >= num_elements:
        return

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    n = offsets // channels
    c = offsets % channels

    # Compute mean over H and W
    sum_val = 0.0
    for h in range(height):
        for w in range(width):
            idx = n * channels * height * width + c * height * width + h * width + w
            val = tl.load(x_ptr + idx)
            sum_val += val

    mean_val = sum_val / (height * width)
    mean_val = mean_val * multiplier
    # Second mean is identity since it's 1x1
    out_val = mean_val

    out_idx = n * channels * 1 * 1 + c * 1 * 1
    tl.store(out_ptr + out_idx, out_val, mask=mask)


def triton_conv_transpose(x, weight, bias, stride, padding, output_padding):
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_h = (in_h - 1) * stride - 2 * padding + kernel_size + output_padding
    out_w = (in_w - 1) * stride - 2 * padding + kernel_size + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    output = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    num_outputs = batch_size * out_channels * out_h * out_w
    BLOCK_SIZE = 128
    grid = lambda meta: ((num_outputs + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    conv_transpose_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, in_h, in_w,
        out_channels, out_h, out_w,
        kernel_size, stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


def triton_multiply_and_double_mean(x, multiplier):
    batch_size, channels, height, width = x.shape
    x = x.contiguous()
    output = torch.empty(batch_size, channels, 1, 1, device=x.device, dtype=x.dtype)

    num_elements = batch_size * channels
    BLOCK_SIZE = 128
    grid = lambda meta: ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    multiply_and_double_mean_kernel[grid](
        x, output,
        batch_size, channels, height, width,
        multiplier,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, x):
        x = triton_conv_transpose(x, self.conv_transpose.weight, self.conv_transpose.bias, self.conv_transpose.stride[0], self.conv_transpose.padding[0], self.conv_transpose.output_padding[0])
        x = triton_multiply_and_double_mean(x, self.multiplier)
        return x