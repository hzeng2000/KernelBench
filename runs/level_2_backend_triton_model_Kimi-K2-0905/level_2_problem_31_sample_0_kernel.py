import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_min_add_scale_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, out_c, out_h, out_w, in_c, k_h, k_w,
    stride_h, stride_w, pad_h, pad_w,
    constant, scale,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_offset = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    h_offset = hw_offset // out_w
    w_offset = hw_offset % out_w

    c_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offset < out_c

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for ic in range(0, in_c):
        for kh in range(0, k_h):
            for kw in range(0, k_w):
                ih = h_offset * stride_h - pad_h + kh
                iw = w_offset * stride_w - pad_w + kw
                mask_hw = (ih >= 0) & (ih < out_h) & (iw >= 0) & (iw < out_w)
                x_idx = ((pid_b * in_c + ic) * out_h + ih) * out_w + iw
                x_val = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)

                w_idx = ((c_offset[:, None] * in_c + ic) * k_h + kh) * k_w + kw
                w_val = tl.load(w_ptr + w_idx, mask=c_mask[:, None], other=0.0)

                acc += x_val[None, :] * w_val

    b_val = tl.load(b_ptr + c_offset, mask=c_mask, other=0.0)
    acc = acc + b_val[:, None]

    acc = tl.where(acc < constant, acc, constant)

    acc = acc * scale

    out_idx = ((pid_b * out_c + c_offset[:, None]) * out_h + h_offset[None, :]) * out_w + w_offset[None, :]
    out_mask = c_mask[:, None] & (hw_offset[None, :] < out_h * out_w)
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.bias.contiguous().view(-1)

        batch, in_c, in_h, in_w = x.shape
        out_c, _, k_h, k_w = w.shape
        stride = self.conv.stride[0]
        pad = self.conv.padding[0]

        out_h = (in_h + 2 * pad - k_h) // stride + 1
        out_w = (in_w + 2 * pad - k_w) // stride + 1

        out = torch.empty((batch, out_c, out_h, out_w), dtype=x.dtype, device=x.device)

        BLOCK_C = 32
        BLOCK_HW = 32

        grid = (batch, (out_c + BLOCK_C - 1) // BLOCK_C, (out_h * out_w + BLOCK_HW - 1) // BLOCK_HW)

        fused_conv_min_add_scale_kernel[grid](
            x, w, b, out,
            batch, out_c, out_h, out_w, in_c, k_h, k_w,
            stride, stride, pad, pad,
            self.constant_value, self.scaling_factor,
            BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW
        )
        return out