import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_sub_tanh_sub_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, h, w, kh, kw,
    stride_h, stride_w, pad_h, pad_w,
    sub1, sub2,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_hw = tl.program_id(2)
    hw = pid_hw * BLOCK_H * BLOCK_W + tl.arange(0, BLOCK_H * BLOCK_W)
    mask_hw = hw < h * w
    oh = hw // w
    ow = hw % w

    acc = tl.zeros([BLOCK_H * BLOCK_W], dtype=tl.float32)
    for ic in range(in_c):
        for kh_ in range(kh):
            for kw_ in range(kw):
                ih = oh * stride_h - pad_h + kh_
                iw = ow * stride_w - pad_w + kw_
                mask_i = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w) & mask_hw
                offs_x = pid_b * in_c * h * w + ic * h * w + ih * w + iw
                x_val = tl.load(x_ptr + offs_x, mask=mask_i, other=0.0)
                offs_w = pid_o * in_c * kh * kw + ic * kh * kw + kh_ * kw + kw_
                w_val = tl.load(w_ptr + offs_w)
                acc += x_val * w_val
    offs_b = pid_o
    b_val = tl.load(b_ptr + offs_b)
    acc += b_val
    acc = acc - sub1
    acc = tl.tanh(acc)
    acc = acc - sub2
    offs_out = pid_b * out_c * h * w + pid_o * h * w + hw
    tl.store(out_ptr + offs_out, acc, mask=mask_hw)


@triton.jit
def avgpool_kernel(
    x_ptr, out_ptr,
    batch, c, ih, iw, oh, ow, kh, kw,
    BLOCK_O: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_o = tl.program_id(2)
    ohw = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    mask_o = ohw < oh * ow
    oh_ = ohw // ow
    ow_ = ohw % ow

    sum_val = tl.zeros([BLOCK_O], dtype=tl.float32)
    for kh_ in range(kh):
        for kw_ in range(kw):
            ih_ = oh_ * kh + kh_
            iw_ = ow_ * kw + kw_
            offs_x = pid_b * c * ih * iw + pid_c * ih * iw + ih_ * iw + iw_
            x_val = tl.load(x_ptr + offs_x, mask=mask_o, other=0.0)
            sum_val += x_val
    avg_val = sum_val / (kh * kw)
    offs_out = pid_b * c * oh * ow + pid_c * oh * ow + ohw
    tl.store(out_ptr + offs_out, avg_val, mask=mask_o)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        w = self.conv.weight
        b = self.conv.bias
        batch, in_c, ih, iw = x.shape
        out_c = w.shape[0]
        kh, kw = w.shape[2], w.shape[3]
        pad_h = self.conv.padding[0]
        pad_w = self.conv.padding[1]
        stride_h = self.conv.stride[0]
        stride_w = self.conv.stride[1]
        oh = (ih + 2 * pad_h - kh) // stride_h + 1
        ow = (iw + 2 * pad_w - kw) // stride_w + 1

        out_conv = torch.empty(batch, out_c, oh, ow, device=x.device, dtype=x.dtype)
        BLOCK_H = 8
        BLOCK_W = 8
        grid = (batch, out_c, (oh * ow + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W))
        fused_conv_sub_tanh_sub_kernel[grid](
            x, w, b, out_conv,
            batch, in_c, out_c, ih, iw, kh, kw,
            stride_h, stride_w, pad_h, pad_w,
            self.subtract1_value, self.subtract2_value,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )

        kh_pool = self.avgpool.kernel_size
        kw_pool = self.avgpool.kernel_size
        oh_pool = oh // kh_pool
        ow_pool = ow // kw_pool
        out_pool = torch.empty(batch, out_c, oh_pool, ow_pool, device=x.device, dtype=x.dtype)
        BLOCK_O = 64
        grid_pool = (batch, out_c, (oh_pool * ow_pool + BLOCK_O - 1) // BLOCK_O)
        avgpool_kernel[grid_pool](
            out_conv, out_pool,
            batch, out_c, oh, ow, oh_pool, ow_pool, kh_pool, kw_pool,
            BLOCK_O=BLOCK_O
        )
        return out_pool