import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, out_c, out_h, out_w, in_c, k_h, k_w,
    scale_factor,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_offset = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    h_offset = hw_offset // out_w
    w_offset = hw_offset % out_w

    c_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_c = c_offset < out_c
    mask_hw = hw_offset < (out_h * out_w)

    acc = tl.full([BLOCK_C, BLOCK_HW], float('-inf'), dtype=tl.float32)

    for ic in range(0, in_c):
        for kh in range(k_h):
            for kw in range(k_w):
                ih = h_offset * stride_h - pad_h + kh
                iw = w_offset * stride_w - pad_w + kw
                mask_in = (ih >= 0) & (ih < out_h) & (iw >= 0) & (iw < out_w) & mask_hw
                x_val = tl.load(x_ptr + pid_b * in_c * out_h * out_w +
                                ic * out_h * out_w +
                                ih * out_w + iw, mask=mask_in, other=float('-inf'))
                w_val = tl.load(w_ptr + c_offset[:, None] * in_c * k_h * k_w +
                                ic * k_h * k_w +
                                kh * k_w + kw, mask=mask_c[:, None], other=0.0)
                acc = tl.where(mask_in[None, :], tl.maximum(acc, x_val[None, :] * w_val), acc)

    acc = acc * scale_factor
    min_val = tl.min(acc, axis=0)
    tl.store(out_ptr + pid_b * out_h * out_w + hw_offset, min_val, mask=mask_hw)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        batch, out_c, out_h, out_w = x.shape
        in_c = self.conv.in_channels
        k_h, k_w = self.conv.kernel_size
        stride_h, stride_w = self.conv.stride
        pad_h, pad_w = self.conv.padding

        out = torch.empty(batch, 1, out_h, out_w, device=x.device, dtype=x.dtype)

        BLOCK_C = 32
        BLOCK_HW = 256

        grid = (batch, (out_c + BLOCK_C - 1) // BLOCK_C, (out_h * out_w + BLOCK_HW - 1) // BLOCK_HW)

        conv_scale_min_kernel[grid](
            x, self.conv.weight, self.conv.bias, out,
            batch, out_c, out_h, out_w, in_c, k_h, k_w,
            self.scale_factor,
            stride_h, stride_w, pad_h, pad_w,
            BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW
        )
        return out