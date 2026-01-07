import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_c, in_h, in_w,
    out_c, out_h, out_w,
    kernel_size, stride, padding, output_padding,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    numel = batch_size * out_c * out_h * out_w
    if pid * BLOCK_SIZE >= numel:
        return

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute output indices
    b_idx = offsets // (out_c * out_h * out_w)
    rem = offsets % (out_c * out_h * out_w)
    c_idx = rem // (out_h * out_w)
    rem = rem % (out_h * out_w)
    h_idx = rem // out_w
    w_idx = rem % out_w

    # Compute input region
    in_h_start = h_idx * stride - padding
    in_w_start = w_idx * stride - padding

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for ic in range(in_c):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                in_h_idx = in_h_start + kh
                in_w_idx = in_w_start + kw
                valid = (in_h_idx >= 0) & (in_h_idx < in_h) & (in_w_idx >= 0) & (in_w_idx < in_w)
                in_offset = ((b_idx * in_c + ic) * in_h + in_h_idx) * in_w + in_w_idx
                w_offset = ((c_idx * in_c + ic) * kernel_size + kh) * kernel_size + kw
                inp = tl.load(input_ptr + in_offset, mask=valid & mask, other=0.0)
                wgt = tl.load(weight_ptr + w_offset)
                acc += inp * wgt

    # Subtract bias and apply tanh
    bias_val = tl.load(bias_ptr + c_idx)
    acc = acc - bias_val
    out_val = tl.tanh(acc)

    tl.store(output_ptr + offsets, out_val, mask=mask)


def triton_conv_transpose_bias_tanh(input, weight, bias, stride, padding, output_padding):
    batch_size, in_c, in_h, in_w = input.shape
    out_c, _, kernel_size, _ = weight.shape
    out_h = (in_h - 1) * stride - 2 * padding + kernel_size + output_padding
    out_w = (in_w - 1) * stride - 2 * padding + kernel_size + output_padding

    output = torch.empty(batch_size, out_c, out_h, out_w, device=input.device, dtype=input.dtype)

    numel = output.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    conv_transpose_kernel[grid](
        input, weight, bias, output,
        batch_size, in_c, in_h, in_w,
        out_c, out_h, out_w,
        kernel_size, stride, padding, output_padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return triton_conv_transpose_bias_tanh(x, self.conv_transpose.weight, self.bias, self.conv_transpose.stride[0], self.conv_transpose.padding[0], self.conv_transpose.output_padding[0])