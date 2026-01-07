import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    kernel_size, stride, padding, output_padding,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    numel = batch_size * out_channels * out_h * out_w
    for i in range(pid, numel, BLOCK_SIZE):
        if i < numel:
            # Compute output location
            n = i // (out_channels * out_h * out_w)
            rem = i % (out_channels * out_h * out_w)
            c_out = rem // (out_h * out_w)
            rem = rem % (out_h * out_w)
            h_out = rem // out_w
            w_out = rem % out_w

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
                            in_idx = n * in_channels * in_h * in_w + c_in * in_h * in_w + h_in * in_w + w_in
                            w_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw
                            acc += tl.load(input_ptr + in_idx) * tl.load(weight_ptr + w_idx)
            out_idx = n * out_channels * out_h * out_w + c_out * out_h * out_w + h_out * out_w + w_out
            tl.store(output_ptr + out_idx, acc)


@triton.jit
def softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    batch_size, channels, height, width,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    numel = batch_size * height * width
    for i in range(pid, numel, BLOCK_SIZE):
        if i < numel:
            n = i // (height * width)
            rem = i % (height * width)
            h = rem // width
            w = rem % width

            # Compute softmax along channel dimension
            max_val = float("-inf")
            for c in range(channels):
                idx = n * channels * height * width + c * height * width + h * width + w
                val = tl.load(x_ptr + idx)
                if val > max_val:
                    max_val = val

            exp_sum = 0.0
            for c in range(channels):
                idx = n * channels * height * width + c * height * width + h * width + w
                val = tl.load(x_ptr + idx)
                exp_val = tl.exp(val - max_val)
                exp_sum += exp_val

            for c in range(channels):
                idx = n * channels * height * width + c * height * width + h * width + w
                val = tl.load(x_ptr + idx)
                exp_val = tl.exp(val - max_val)
                softmax_val = exp_val / exp_sum

                # Add bias, scale, sigmoid
                bias_val = tl.load(bias_ptr + c)
                out_val = softmax_val + bias_val
                out_val = out_val * scaling_factor
                out_val = 1.0 / (1.0 + tl.exp(-out_val))
                tl.store(out_ptr + idx, out_val)


def triton_conv_transpose2d(input, weight, stride, padding, output_padding):
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_h = (in_h - 1) * stride - 2 * padding + kernel_size + output_padding
    out_w = (in_w - 1) * stride - 2 * padding + kernel_size + output_padding
    output = torch.empty(batch_size, out_channels, out_h, out_w, device=input.device, dtype=input.dtype)
    numel = batch_size * out_channels * out_h * out_w
    BLOCK_SIZE = 128
    grid = lambda meta: ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    conv_transpose_kernel[grid](
        input, weight, output,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_size, stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


def triton_softmax_bias_scale_sigmoid(x, bias, scaling_factor):
    batch_size, channels, height, width = x.shape
    output = torch.empty_like(x)
    numel = batch_size * height * width
    BLOCK_SIZE = 128
    grid = lambda meta: ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    softmax_bias_scale_sigmoid_kernel[grid](
        x, bias, output,
        batch_size, channels, height, width,
        scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = triton_conv_transpose2d(x, self.weight, self.stride, self.padding, self.output_padding)
        x = triton_softmax_bias_scale_sigmoid(x, self.bias, self.scaling_factor)
        return x