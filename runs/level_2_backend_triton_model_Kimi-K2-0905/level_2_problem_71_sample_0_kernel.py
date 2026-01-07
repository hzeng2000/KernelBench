import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_div_leaky_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, h, w, k,
    stride_h, stride_w, pad_h, pad_w,
    divisor, negative_slope,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    batch_offs = pid_b * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    out_c_offs = pid_oc * BLOCK_OUT_C + tl.arange(0, BLOCK_OUT_C)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    batch_mask = batch_offs < batch
    out_c_mask = out_c_offs < out_c
    h_mask = h_offs < h
    w_mask = w_offs < w

    out_h = (h + 2 * pad_h - k) // stride_h + 1
    out_w = (w + 2 * pad_w - k) // stride_w + 1

    for ob in range(BLOCK_BATCH):
        if batch_mask[ob]:
            for oc in range(BLOCK_OUT_C):
                if out_c_mask[oc]:
                    for oh in range(BLOCK_H):
                        if h_mask[oh]:
                            for ow in range(BLOCK_W):
                                if w_mask[ow]:
                                    acc = 0.0
                                    for ic in range(in_c):
                                        for kh in range(k):
                                            for kw in range(k):
                                                ih = oh * stride_h - pad_h + kh
                                                iw = ow * stride_w - pad_w + kw
                                                if ih >= 0 and ih < h and iw >= 0 and iw < w:
                                                    x_val = tl.load(x_ptr + ((batch_offs[ob] * in_c + ic) * h + ih) * w + iw)
                                                    w_val = tl.load(w_ptr + ((out_c_offs[oc] * in_c + ic) * k + kh) * k + kw)
                                                    acc += x_val * w_val
                                    b_val = tl.load(b_ptr + out_c_offs[oc])
                                    acc = (acc + b_val) / divisor
                                    out_val = tl.where(acc > 0, acc, acc * negative_slope)
                                    out_idx = ((batch_offs[ob] * out_c + out_c_offs[oc]) * out_h + h_offs[oh]) * out_w + w_offs[ow]
                                    tl.store(out_ptr + out_idx, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight
        b = self.conv.bias
        batch, in_c, h, w = x.shape
        out_c = w.shape[0]
        k = w.shape[2]
        out_h = h
        out_w = w
        out = torch.empty(batch, out_c, out_h, out_w, dtype=x.dtype, device=x.device)

        BLOCK_BATCH = 1
        BLOCK_OUT_C = 1
        BLOCK_H = 16
        BLOCK_W = 16

        grid = (
            (batch + BLOCK_BATCH - 1) // BLOCK_BATCH,
            (out_c + BLOCK_OUT_C - 1) // BLOCK_OUT_C,
            (out_h + BLOCK_H - 1) // BLOCK_H,
            (out_w + BLOCK_W - 1) // BLOCK_W,
        )

        conv_div_leaky_relu_kernel[grid](
            x, w, b, out,
            batch, in_c, out_c, h, w, k,
            1, 1, 0, 0,
            self.divisor, 0.01,
            BLOCK_BATCH, BLOCK_OUT_C, BLOCK_H, BLOCK_W,
        )
        return out