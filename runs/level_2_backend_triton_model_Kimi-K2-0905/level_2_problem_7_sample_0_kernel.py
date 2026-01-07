import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv3d_bias_activations_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    depth, height, width,
    kernel_size,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_depth, out_height, out_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = batch_size * out_channels * out_depth * out_height * out_width
    if pid * BLOCK_SIZE >= n_elements:
        return

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute 5D indices from linear offset
    n = offsets // (out_channels * out_depth * out_height * out_width)
    remainder = offsets % (out_channels * out_depth * out_height * out_width)
    c_out = remainder // (out_depth * out_height * out_width)
    remainder = remainder % (out_depth * out_height * out_width)
    d_out = remainder // (out_height * out_width)
    remainder = remainder % (out_height * out_width)
    h_out = remainder // out_width
    w_out = remainder % out_width

    # Compute input start indices
    d_start = d_out * stride_d - pad_d
    h_start = h_out * stride_h - pad_h
    w_start = w_out * stride_w - pad_w

    acc = 0.0
    for c_in in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    d_in = d_start + kd
                    h_in = h_start + kh
                    w_in = w_start + kw
                    if 0 <= d_in < depth and 0 <= h_in < height and 0 <= w_in < width:
                        x_idx = n * in_channels * depth * height * width + \
                                c_in * depth * height * width + \
                                d_in * height * width + \
                                h_in * width + w_in
                        w_idx = c_out * in_channels * kernel_size * kernel_size * kernel_size + \
                                c_in * kernel_size * kernel_size * kernel_size + \
                                kd * kernel_size * kernel_size + \
                                kh * kernel_size + kw
                        x_val = tl.load(x_ptr + x_idx)
                        w_val = tl.load(w_ptr + w_idx)
                        acc += x_val * w_val

    # Add bias
    bias_val = tl.load(b_ptr + c_out)
    acc += bias_val

    # Apply activations: ReLU, LeakyReLU, GELU, Sigmoid
    acc = tl.maximum(acc, 0.0)  # ReLU
    acc = tl.where(acc > 0, acc, acc * 0.01)  # LeakyReLU
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gelu_approx = 0.5 * acc * (1.0 + tl.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
    acc = gelu_approx
    # Sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))

    out_idx = n * out_channels * out_depth * out_height * out_width + \
              c_out * out_depth * out_height * out_width + \
              d_out * out_height * out_width + \
              h_out * out_width + w_out
    tl.store(out_ptr + out_idx, acc, mask=mask)


def triton_conv3d_fused(x, weight, bias, kernel_size, stride=1, padding=0):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels = weight.shape[0]
    kernel_size = kernel_size
    stride_d = stride_h = stride_w = stride
    pad_d = pad_h = pad_w = padding

    out_depth = (depth + 2 * pad_d - kernel_size) // stride_d + 1
    out_height = (height + 2 * pad_h - kernel_size) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_size) // stride_w + 1

    out = torch.empty(batch_size, out_channels, out_depth, out_height, out_width,
                      dtype=x.dtype, device=x.device)

    n_elements = batch_size * out_channels * out_depth * out_height * out_width
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_conv3d_bias_activations_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_depth, out_height, out_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return triton_conv3d_fused(x, self.weight, self.bias, kernel_size=3, stride=1, padding=0)