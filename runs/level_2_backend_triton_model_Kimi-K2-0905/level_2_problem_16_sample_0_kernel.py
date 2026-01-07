import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_mish_add_hardtanh_scale_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, in_h, in_w, out_h, out_w,
    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w,
    add_val, scale,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    pid_h = pid_hw // ((out_h + BLOCK_H - 1) // BLOCK_H)
    pid_w = pid_hw % ((out_w + BLOCK_W - 1) // BLOCK_W)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h_offs < out_h
    w_mask = w_offs < out_w

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for ic in range(in_c):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                ih = (h_offs + pad_h - kh * 1 - out_pad_h * 1) // stride_h
                iw = (w_offs + pad_w - kw * 1 - out_pad_w * 1) // stride_w
                mask = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w) & h_mask & w_mask
                ih = tl.where(ih >= 0, ih, 0)
                iw = tl.where(iw >= 0, iw, 0)
                x_idx = pid_b * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw
                x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                w_idx = pid_c * in_c * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_c)
        acc += b_val

    # Mish activation
    mish_out = acc * tl.tanh(tl.log(1 + tl.exp(acc)))

    # Add value
    mish_out += add_val

    # Hardtanh activation
    hardtanh_out = tl.where(mish_out > 1.0, 1.0, tl.where(mish_out < -1.0, -1.0, mish_out))

    # Scale
    final_out = hardtanh_out * scale

    out_idx_base = pid_b * out_c * out_h * out_w + pid_c * out_h * out_w
    out_idx = out_idx_base + (h_offs[:, None] * out_w + w_offs[None, :])
    out_mask = h_mask[:, None] & w_mask[None, :]
    tl.store(out_ptr + out_idx, final_out, mask=out_mask)


def triton_conv_transpose_mish_add_hardtanh_scale(x, weight, bias, stride, padding, output_padding, add_value, scale):
    batch, in_c, in_h, in_w = x.shape
    out_c, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_pad_h, out_pad_w = output_padding

    out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
    out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w

    out = torch.empty(batch, out_c, out_h, out_w, dtype=x.dtype, device=x.device)

    BLOCK_H = 8
    BLOCK_W = 8

    grid = (
        batch,
        out_c,
        ((out_h + BLOCK_H - 1) // BLOCK_H) * ((out_w + BLOCK_W - 1) // BLOCK_W),
    )

    conv_transpose_mish_add_hardtanh_scale_kernel[grid](
        x, weight, bias, out,
        batch, in_c, out_c, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w,
        add_value, scale,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        return triton_conv_transpose_mish_add_hardtanh_scale(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride,
            self.conv_transpose.padding,
            self.conv_transpose.output_padding,
            self.add_value,
            self.scale
        )