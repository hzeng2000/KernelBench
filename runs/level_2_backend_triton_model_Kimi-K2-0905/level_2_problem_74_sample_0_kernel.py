import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_leaky_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, out_c, out_d, out_h, out_w,
    in_c, in_d, in_h, in_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_pad_d, out_pad_h, out_pad_w,
    negative_slope,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)
    pid_d = pid_dhw // (out_h * out_w)
    pid_hw = pid_dhw % (out_h * out_w)
    pid_h = pid_hw // out_w
    pid_w = pid_hw % out_w

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    h_offs = hw_offs // out_w
    w_offs = hw_offs % out_w

    mask_c = c_offs < out_c
    mask_d = d_offs < out_d
    mask_h = h_offs < out_h
    mask_w = w_offs < out_w
    mask = mask_c[:, None, None, None] & mask_d[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :]

    acc = tl.zeros((BLOCK_C, BLOCK_D, BLOCK_HW, BLOCK_HW), dtype=tl.float32)

    for ic in range(in_c):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    in_d = (d_offs - kd + pad_d) // stride_d
                    in_h = (h_offs - kh + pad_h) // stride_h
                    in_w = (w_offs - kw + pad_w) // stride_w
                    valid_d = (in_d >= 0) & (in_d < in_d) & ((d_offs - kd + pad_d) % stride_d == 0)
                    valid_h = (in_h >= 0) & (in_h < in_h) & ((h_offs - kh + pad_h) % stride_h == 0)
                    valid_w = (in_w >= 0) & (in_w < in_w) & ((w_offs - kw + pad_w) % stride_w == 0)
                    valid = valid_d[None, :, None, None] & valid_h[None, None, :, None] & valid_w[None, None, None, :]

                    x_idx = pid_b * in_c * in_d * in_h * in_w + ic * in_d * in_h * in_w + in_d * in_h * in_w + in_h * in_w + in_w
                    w_idx = pid_c * in_c * kernel_d * kernel_h * kernel_w + ic * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw

                    x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
                    w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)

                    acc += x_val * w_val

    if b_ptr:
        b_val = tl.load(b_ptr + c_offs, mask=mask_c, other=0.0)
        acc += b_val[:, None, None, None]

    out = tl.where(acc > 0, acc, acc * negative_slope)

    out_idx = pid_b * out_c * out_d * out_h * out_w + c_offs[:, None, None, None] * out_d * out_h * out_w + d_offs[None, :, None, None] * out_h * out_w + h_offs[None, None, :, None] * out_w + w_offs[None, None, None, :]
    tl.store(out_ptr + out_idx, out, mask=mask)


@triton.jit
def fused_multiply_leaky_relu_maxpool_kernel(
    x_ptr, m_ptr, out_ptr,
    batch, channels, depth, height, width,
    out_depth, out_height, out_width,
    negative_slope,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)
    pid_d = pid_dhw // (out_height * out_width)
    pid_hw = pid_dhw % (out_height * out_width)
    pid_h = pid_hw // out_width
    pid_w = pid_hw % out_width

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    h_offs = hw_offs // out_width
    w_offs = hw_offs % out_width

    mask_c = c_offs < channels
    mask_d = d_offs < out_depth
    mask_h = h_offs < out_height
    mask_w = w_offs < out_width
    mask = mask_c[:, None, None, None] & mask_d[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :]

    max_val = float('-inf')

    for kd in range(2):
        for kh in range(2):
            for kw in range(2):
                in_d = d_offs * 2 + kd
                in_h = h_offs * 2 + kh
                in_w = w_offs * 2 + kw

                in_mask_d = in_d < depth
                in_mask_h = in_h < height
                in_mask_w = in_w < width
                in_mask = mask_c[:, None, None, None] & in_mask_d[None, :, None, None] & in_mask_h[None, None, :, None] & in_mask_w[None, None, None, :]

                x_idx = pid_b * channels * depth * height * width + c_offs[:, None, None, None] * depth * height * width + in_d[None, :, None, None] * height * width + in_h[None, None, :, None] * width + in_w[None, None, None, :]
                x_val = tl.load(x_ptr + x_idx, mask=in_mask, other=float('-inf'))

                m_val = tl.load(m_ptr + c_offs, mask=mask_c, other=1.0)
                x_val = x_val * m_val[:, None, None, None]

                out_val = tl.where(x_val > 0, x_val, x_val * negative_slope)
                max_val = tl.maximum(max_val, out_val)

    out_idx = pid_b * channels * out_depth * out_height * out_width + c_offs[:, None, None, None] * out_depth * out_height * out_width + d_offs[None, :, None, None] * out_height * out_width + h_offs[None, None, :, None] * out_width + w_offs[None, None, None, :]
    tl.store(out_ptr + out_idx, max_val, mask=mask)


def triton_conv_transpose_leaky_relu(x, weight, bias, stride, padding, output_padding, negative_slope):
    batch, in_c, in_d, in_h, in_w = x.shape
    out_c, _, kernel_d, kernel_h, kernel_w = weight.shape
    out_d = (in_d - 1) * stride - 2 * padding + kernel_d + output_padding
    out_h = (in_h - 1) * stride - 2 * padding + kernel_h + output_padding
    out_w = (in_w - 1) * stride - 2 * padding + kernel_w + output_padding

    out = torch.empty(batch, out_c, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

    BLOCK_C = 4
    BLOCK_D = 4
    BLOCK_HW = 4

    grid = (batch, (out_c + BLOCK_C - 1) // BLOCK_C, (out_d * out_h * out_w + BLOCK_D * BLOCK_HW * BLOCK_HW - 1) // (BLOCK_D * BLOCK_HW * BLOCK_HW))

    fused_transpose_leaky_relu_kernel[grid](
        x, weight, bias, out,
        batch, out_c, out_d, out_h, out_w,
        in_c, in_d, in_h, in_w,
        kernel_d, kernel_h, kernel_w,
        stride, stride, stride,
        padding, padding, padding,
        output_padding, output_padding, output_padding,
        negative_slope,
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, BLOCK_HW=BLOCK_HW
    )

    return out


def triton_multiply_leaky_relu_maxpool(x, multiplier, negative_slope):
    batch, channels, depth, height, width = x.shape
    out_depth = depth // 2
    out_height = height // 2
    out_width = width // 2

    out = torch.empty(batch, channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

    BLOCK_C = 4
    BLOCK_D = 4
    BLOCK_HW = 4

    grid = (batch, (channels + BLOCK_C - 1) // BLOCK_C, (out_depth * out_height * out_width + BLOCK_D * BLOCK_HW * BLOCK_HW - 1) // (BLOCK_D * BLOCK_HW * BLOCK_HW))

    fused_multiply_leaky_relu_maxpool_kernel[grid](
        x, multiplier, out,
        batch, channels, depth, height, width,
        out_depth, out_height, out_width,
        negative_slope,
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, BLOCK_HW=BLOCK_HW
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.negative_slope = 0.2

    def forward(self, x):
        x = triton_conv_transpose_leaky_relu(x, self.conv_transpose.weight, self.conv_transpose.bias, self.conv_transpose.stride[0], self.conv_transpose.padding[0], self.conv_transpose.output_padding[0], self.negative_slope)
        x = triton_multiply_leaky_relu_maxpool(x, self.multiplier, self.negative_slope)
        return x