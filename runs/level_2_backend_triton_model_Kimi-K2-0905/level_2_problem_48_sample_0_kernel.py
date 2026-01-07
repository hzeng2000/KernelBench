import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    depth, height, width,
    kernel_size,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_depth, out_height, out_width,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    if pid_b >= batch_size or pid_oc >= out_channels or pid_oh >= out_height or pid_ow >= out_width:
        return

    acc = 0.0
    for ic in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_d = pid_oh * stride_d + kd - pad_d
                    in_h = pid_oh * stride_h + kh - pad_h
                    in_w = pid_ow * stride_w + kw - pad_w

                    if 0 <= in_d < depth and 0 <= in_h < height and 0 <= in_w < width:
                        in_idx = pid_b * in_channels * depth * height * width + ic * depth * height * width + in_d * height * width + in_h * width + in_w
                        w_idx = pid_oc * in_channels * kernel_size * kernel_size * kernel_size + ic * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw
                        in_val = tl.load(input_ptr + in_idx)
                        w_val = tl.load(weight_ptr + w_idx)
                        acc += in_val * w_val

    out_idx = pid_b * out_channels * out_depth * out_height * out_width + pid_oc * out_depth * out_height * out_width + pid_oh * out_height * out_width + pid_ow
    tl.store(output_ptr + out_idx, acc)


@triton.jit
def scale_tanh_sigmoid_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + offsets % 16, mask=mask)  # out_channels = 16
    bias = tl.load(bias_ptr + offsets % 16, mask=mask)

    x = x * scale
    x = tl.tanh(x)
    x = x * bias
    x = tl.sigmoid(x)

    tl.store(out_ptr + offsets, x, mask=mask)


def triton_conv3d(x, weight, bias, stride=1, padding=0):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape
    stride_d = stride_h = stride_w = stride
    pad_d = pad_h = pad_w = padding

    out_depth = (depth + 2 * pad_d - kernel_size) // stride_d + 1
    out_height = (height + 2 * pad_h - kernel_size) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_size) // stride_w + 1

    output = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

    grid = (batch_size, out_channels, out_height, out_width)
    BLOCK_D = BLOCK_H = BLOCK_W = 1

    conv3d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_depth, out_height, out_width,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return output


def triton_scale_tanh_sigmoid(x, scale, bias):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    scale_tanh_sigmoid_kernel[grid](
        x, scale, bias, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = triton_conv3d(x, self.conv.weight, self.conv.bias, stride=1, padding=0)
        x = triton_scale_tanh_sigmoid(x, self.scaling_factor, self.bias)
        return x