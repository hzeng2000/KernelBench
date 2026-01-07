import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_mul_leaky_gelu_kernel(
    x_ptr, w_ptr, b_ptr, m_ptr, out_ptr,
    batch, out_c, out_h, out_w, in_c, k_h, k_w,
    stride_h, stride_w, pad_h, pad_w,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    c_start = pid_c * BLOCK_C
    hw_start = pid_hw * BLOCK_HW

    c_offs = c_start + tl.arange(0, BLOCK_C)
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)

    c_mask = c_offs < out_c
    hw_mask = hw_offs < (out_h * out_w)

    h_offs = hw_offs // out_w
    w_offs = hw_offs % out_w

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for ic in range(in_c):
        for kh in range(k_h):
            for kw in range(k_w):
                ih = h_offs * stride_h - pad_h + kh
                iw = w_offs * stride_w - pad_w + kw
                in_bounds = (ih >= 0) & (ih < out_h) & (iw >= 0) & (iw < out_w)
                x_idx = ((pid_b * in_c + ic) * (out_h + 2 * pad_h) + ih) * (out_w + 2 * pad_w) + iw
                x_val = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)

                w_idx = ((c_offs[:, None] * in_c + ic) * k_h + kh) * k_w + kw
                w_val = tl.load(w_ptr + w_idx, mask=c_mask[:, None], other=0.0)

                acc += w_val * x_val

    b_val = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0)
    acc = acc + b_val[:, None]

    m_val = tl.load(m_ptr + c_offs, mask=c_mask, other=0.0)
    acc = acc * m_val[:, None]

    acc = tl.where(acc > 0, acc, 0.01 * acc)

    acc = 0.5 * acc * (1.0 + tl.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

    out_idx = ((pid_b * out_c + c_offs[:, None]) * out_h + h_offs[None, :]) * out_w + w_offs[None, :]
    tl.store(out_ptr + out_idx, acc, mask=c_mask[:, None] & hw_mask[None, :])


def triton_fused_conv_mul_leaky_gelu(x, w, b, m, stride=1, padding=1):
    batch, in_c, in_h, in_w = x.shape
    out_c, _, k_h, k_w = w.shape
    stride_h = stride_w = stride
    pad_h = pad_w = padding

    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1

    out = torch.empty(batch, out_c, out_h, out_w, device=x.device, dtype=x.dtype)

    BLOCK_C = 32
    BLOCK_HW = 32

    grid = (batch, (out_c + BLOCK_C - 1) // BLOCK_C, (out_h * out_w + BLOCK_HW - 1) // BLOCK_HW)

    fused_conv_mul_leaky_gelu_kernel[grid](
        x, w, b, m, out,
        batch, out_c, out_h, out_w, in_c, k_h, k_w,
        stride_h, stride_w, pad_h, pad_w,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x):
        return triton_fused_conv_mul_leaky_gelu(
            x, self.conv.weight, self.conv.bias, self.multiplier
        )