import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_threads = batch_size * out_channels * out_d * out_h * out_w
    if pid >= total_threads:
        return

    # Compute output indices
    tmp = pid
    b = tmp // (out_channels * out_d * out_h * out_w)
    tmp %= (out_channels * out_d * out_h * out_w)
    c_out = tmp // (out_d * out_h * out_w)
    tmp %= (out_d * out_h * out_w)
    d_out = tmp // (out_h * out_w)
    tmp %= (out_h * out_w)
    h_out = tmp // out_w
    w_out = tmp % out_w

    # Compute input start indices
    d_start = d_out * stride_d - pad_d
    h_start = h_out * stride_h - pad_h
    w_start = w_out * stride_w - pad_w

    acc = 0.0
    for c_in in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    d_in = d_start + kd
                    h_in = h_start + kh
                    w_in = w_start + kw
                    if d_in >= 0 and d_in < in_d and h_in >= 0 and h_in < in_h and w_in >= 0 and w_in < in_w:
                        in_idx = ((b * in_channels + c_in) * in_d + d_in) * in_h * in_w + h_in * in_w + w_in
                        w_idx = ((c_out * in_channels + c_in) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw
                        acc += tl.load(input_ptr + in_idx) * tl.load(weight_ptr + w_idx)
    out_idx = ((b * out_channels + c_out) * out_d + d_out) * out_h * out_w + h_out * out_w + w_out
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + c_out)
    tl.store(output_ptr + out_idx, acc)


@triton.jit
def fused_maxpool_sum_kernel(
    input_ptr, output_ptr,
    batch_size, channels, in_d, in_h, in_w,
    out_d1, out_h1, out_w1,
    out_d2, out_h2, out_w2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_threads = batch_size * out_d2 * out_h2 * out_w2
    if pid >= total_threads:
        return

    tmp = pid
    b = tmp // (out_d2 * out_h2 * out_w2)
    tmp %= (out_d2 * out_h2 * out_w2)
    d2 = tmp // (out_h2 * out_w2)
    tmp %= (out_h2 * out_w2)
    h2 = tmp // out_w2
    w2 = tmp % out_w2

    # MaxPool1: kernel=2, stride=2
    d1 = d2 * 2
    h1 = h2 * 2
    w1 = w2 * 2

    max_val = float('-inf')
    for c in range(channels):
        for kd in range(2):
            for kh in range(2):
                for kw in range(2):
                    in_idx = ((b * channels + c) * in_d + (d1 + kd)) * in_h * in_w + (h1 + kh) * in_w + (w1 + kw)
                    val = tl.load(input_ptr + in_idx)
                    if val > max_val:
                        max_val = val

    # MaxPool2: kernel=3, stride=3
    d2_3 = d2 // 3
    h2_3 = h2 // 3
    w2_3 = w2 // 3

    max_val2 = float('-inf')
    for c in range(channels):
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    d3 = d2_3 * 3 + kd
                    h3 = h2_3 * 3 + kh
                    w3 = w2_3 * 3 + kw
                    if d3 < out_d2 and h3 < out_h2 and w3 < out_w2:
                        in_idx = ((b * channels + c) * out_d2 + d3) * out_h2 * out_w2 + h3 * out_w2 + w3
                        val = tl.load(input_ptr + in_idx)
                        if val > max_val2:
                            max_val2 = val

    # Sum over channels, keepdim=True
    out_idx = (b * 1 * out_d2 * out_h2 * out_w2) + d2 * out_h2 * out_w2 + h2 * out_w2 + w2
    tl.store(output_ptr + out_idx, max_val2)


def triton_conv_transpose3d(input, weight, bias, stride, padding):
    batch_size, in_channels, in_d, in_h, in_w = input.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d
    out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h
    out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w

    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=input.dtype, device=input.device)

    total_threads = batch_size * out_channels * out_d * out_h * out_w
    BLOCK_SIZE = 128
    grid = lambda meta: ((total_threads + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    conv_transpose_kernel[grid](
        input, weight, bias, output,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


def triton_fused_maxpool_sum(input):
    batch_size, channels, in_d, in_h, in_w = input.shape

    # MaxPool1: kernel=2, stride=2
    out_d1 = in_d // 2
    out_h1 = in_h // 2
    out_w1 = in_w // 2

    # MaxPool2: kernel=3, stride=3
    out_d2 = out_d1 // 3
    out_h2 = out_h1 // 3
    out_w2 = out_w1 // 3

    output = torch.empty(batch_size, 1, out_d2, out_h2, out_w2, dtype=input.dtype, device=input.device)

    total_threads = batch_size * out_d2 * out_h2 * out_w2
    BLOCK_SIZE = 128
    grid = lambda meta: ((total_threads + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_maxpool_sum_kernel[grid](
        input, output,
        batch_size, channels, in_d, in_h, in_w,
        out_d1, out_h1, out_w1,
        out_d2, out_h2, out_w2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = triton_conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding)
        x = triton_fused_maxpool_sum(x)
        return x