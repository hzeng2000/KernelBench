import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_hardswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    hw = height * width
    nhw = batch_size * hw
    num_blocks = (nhw + BLOCK_SIZE - 1) // BLOCK_SIZE

    if pid >= num_blocks:
        return

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nhw

    # Compute n, c_out, h, w from offset
    n = offsets // hw
    rem = offsets % hw
    h = rem // width
    w = rem % width

    # Compute output channel index
    c_out = tl.arange(0, out_channels)

    acc = tl.zeros((BLOCK_SIZE, out_channels), dtype=tl.float32)

    # Convolution loop
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h_in = h * stride - padding + kh
            w_in = w * stride - padding + kw
            valid = (h_in >= 0) & (h_in < height) & (w_in >= 0) & (w_in < width)

            for c_in in range(in_channels):
                x_idx = n * in_channels * height * width + c_in * height * width + h_in * width + w_in
                x_val = tl.load(x_ptr + x_idx, mask=mask & valid, other=0.0)

                w_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw
                w_val = tl.load(w_ptr + w_idx)

                acc += x_val[:, None] * w_val[None, :]

    # Add bias
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_out)
        acc += b_val[None, :]

    # HardSwish
    out = acc * tl.maximum(0.0, tl.minimum(3.0, acc + 3.0)) / 6.0

    # ReLU
    out = tl.maximum(0.0, out)

    # Store output
    out_idx = n[:, None] * out_channels * height * width + c_out[None, :] * height * width + h[:, None] * width + w[:, None]
    tl.store(out_ptr + out_idx, out, mask=mask[:, None])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2

    def forward(self, x):
        batch_size, _, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = torch.empty(batch_size, self.out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

        n_elements = batch_size * out_height * out_width
        BLOCK_SIZE = 128

        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        fused_conv_hardswish_relu_kernel[grid](
            x, self.conv.weight, self.conv.bias, output,
            batch_size, self.in_channels, self.out_channels, out_height, out_width,
            self.kernel_size, self.stride, self.padding,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return output


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]