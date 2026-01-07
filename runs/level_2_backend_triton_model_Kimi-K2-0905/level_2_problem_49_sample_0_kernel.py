import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose3d_softmax_sigmoid_kernel(
    out_ptr, in_ptr, weight_ptr, bias_ptr,
    batch_size, in_channels, out_channels,
    out_d, out_h, out_w,
    in_d, in_h, in_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    # Compute output spatial indices
    d = pid_dhw // (out_h * out_w)
    hw = pid_dhw % (out_h * out_w)
    h = hw // out_w
    w = hw % out_w

    # Compute input spatial range
    start_d = d * stride_d - pad_d
    start_h = h * stride_h - pad_h
    start_w = w * stride_w - pad_w

    # Accumulator
    acc = 0.0
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + pid_c)

    # Loop over input channels and kernel
    for ic in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    in_d_idx = start_d + kd
                    in_h_idx = start_h + kh
                    in_w_idx = start_w + kw

                    # Check bounds
                    if in_d_idx >= 0 and in_d_idx < in_d and \
                       in_h_idx >= 0 and in_h_idx < in_h and \
                       in_w_idx >= 0 and in_w_idx < in_w:
                        # Load input
                        in_offset = pid_b * in_channels * in_d * in_h * in_w + \
                                    ic * in_d * in_h * in_w + \
                                    in_d_idx * in_h * in_w + \
                                    in_h_idx * in_w + \
                                    in_w_idx
                        in_val = tl.load(in_ptr + in_offset)

                        # Load weight
                        weight_offset = pid_c * in_channels * kernel_d * kernel_h * kernel_w + \
                                        ic * kernel_d * kernel_h * kernel_w + \
                                        kd * kernel_h * kernel_w + \
                                        kh * kernel_w + \
                                        kw
                        weight_val = tl.load(weight_ptr + weight_offset)

                        acc += in_val * weight_val

    # Store output before activation
    out_offset = pid_b * out_channels * out_d * out_h * out_w + \
                 pid_c * out_d * out_h * out_w + \
                 d * out_h * out_w + \
                 h * out_w + \
                 w
    tl.store(out_ptr + out_offset, acc)

    # Softmax along channel dimension
    # First pass: compute max for numerical stability
    max_val = float('-inf')
    for c in range(out_channels):
        offset = pid_b * out_channels * out_d * out_h * out_w + \
                 c * out_d * out_h * out_w + \
                 d * out_h * out_w + \
                 h * out_w + \
                 w
        val = tl.load(out_ptr + offset)
        max_val = tl.maximum(max_val, val)

    # Second pass: compute exp and sum
    sum_exp = 0.0
    for c in range(out_channels):
        offset = pid_b * out_channels * out_d * out_h * out_w + \
                 c * out_d * out_h * out_w + \
                 d * out_h * out_w + \
                 h * out_w + \
                 w
        val = tl.load(out_ptr + offset)
        exp_val = tl.exp(val - max_val)
        tl.store(out_ptr + offset, exp_val)
        sum_exp += exp_val

    # Third pass: normalize
    for c in range(out_channels):
        offset = pid_b * out_channels * out_d * out_h * out_w + \
                 c * out_d * out_h * out_w + \
                 d * out_h * out_w + \
                 h * out_w + \
                 w
        val = tl.load(out_ptr + offset)
        softmax_val = val / sum_exp
        sigmoid_val = 1.0 / (1.0 + tl.exp(-softmax_val))
        tl.store(out_ptr + offset, sigmoid_val)


def triton_transpose3d_softmax_sigmoid(x, weight, bias, stride, padding, output_padding):
    batch_size, in_channels, in_d, in_h, in_w = x.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape

    # Compute output dimensions
    out_d = (in_d - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0]
    out_h = (in_h - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1]
    out_w = (in_w - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2]

    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

    # Grid dimensions
    grid = (batch_size, out_channels, out_d * out_h * out_w)

    # Launch kernel
    fused_transpose3d_softmax_sigmoid_kernel[grid](
        out, x, weight, bias,
        batch_size, in_channels, out_channels,
        out_d, out_h, out_w,
        in_d, in_h, in_w,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        BLOCK_D=4, BLOCK_H=8, BLOCK_W=8, BLOCK_C=32,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.weight = self.conv_transpose.weight
        self.bias = self.conv_transpose.bias
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        return triton_transpose3d_softmax_sigmoid(x, self.weight, self.bias, self.stride, self.padding, self.output_padding)