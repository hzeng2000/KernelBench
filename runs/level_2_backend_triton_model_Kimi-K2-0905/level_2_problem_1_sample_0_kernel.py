import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, h, w, k,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    out_h: tl.constexpr, out_w: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_start = pid_oc * BLOCK_C
    hw_start = pid_hw * BLOCK_HW

    oc_offs = oc_start + tl.arange(0, BLOCK_C)
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)

    mask_oc = oc_offs < out_c
    mask_hw = hw_offs < (out_h * out_w)

    oh = hw_offs // out_w
    ow = hw_offs % out_w

    ih_base = oh * stride_h - pad_h
    iw_base = ow * stride_w - pad_w

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for ic in range(in_c):
        for kh in range(k):
            for kw in range(k):
                ih = ih_base + kh
                iw = iw_base + kw
                mask_i = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
                x_idx = (
                    pid_b * in_c * h * w +
                    ic * h * w +
                    ih * w +
                    iw
                )
                x_val = tl.load(x_ptr + x_idx, mask=mask_i & mask_hw[:, None], other=0.0)

                w_idx = (
                    oc_offs[:, None] * in_c * k * k +
                    ic * k * k +
                    kh * k +
                    kw
                )
                w_val = tl.load(w_ptr + w_idx, mask=mask_oc[:, None], other=0.0)

                acc += w_val * x_val

    b_val = tl.load(b_ptr + oc_offs, mask=mask_oc, other=0.0)
    out_val = tl.maximum(acc + b_val[:, None], 0.0)

    out_idx = (
        pid_b * out_c * out_h * out_w +
        oc_offs[:, None] * out_h * out_w +
        hw_offs[None, :]
    )
    tl.store(out_ptr + out_idx, out_val, mask=mask_oc[:, None] & mask_hw[None, :])


def triton_conv_relu_bias(x, w, b, stride=1, padding=1):
    batch, in_c, h, w = x.shape
    out_c, _, k, _ = w.shape
    stride_h = stride_w = stride
    pad_h = pad_w = padding
    out_h = (h + 2 * pad_h - k) // stride_h + 1
    out_w = (w + 2 * pad_w - k) // stride_w + 1

    out = torch.empty((batch, out_c, out_h, out_w), dtype=x.dtype, device=x.device)

    BLOCK_C = 32
    BLOCK_HW = 32

    grid = (
        batch,
        (out_c + BLOCK_C - 1) // BLOCK_C,
        (out_h * out_w + BLOCK_HW - 1) // BLOCK_HW
    )

    conv_relu_bias_kernel[grid](
        x, w, b, out,
        batch, in_c, out_c, h, w, k,
        stride_h, stride_w, pad_h, pad_w, out_h, out_w,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        w = self.conv.weight
        b = self.bias.squeeze()
        return triton_conv_relu_bias(x, w, b)