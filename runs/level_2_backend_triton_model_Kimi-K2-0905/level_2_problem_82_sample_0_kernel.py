import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_max_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, h, w,
    kernel_size, stride, padding,
    scaling_factor,
    pool_kernel, pool_stride,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUTC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    batch_offs = pid_b * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    outc_offs = pid_oc * BLOCK_OUTC + tl.arange(0, BLOCK_OUTC)
    oh_offs = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_offs = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)

    batch_mask = batch_offs < batch
    outc_mask = outc_offs < out_c
    oh_mask = oh_offs < (h // pool_stride)
    ow_mask = ow_offs < (w // pool_stride)

    pool_h = oh_offs * pool_stride
    pool_w = ow_offs * pool_stride

    max_val = float('-inf')

    for ph in range(pool_kernel):
        for pw in range(pool_kernel):
            ih = pool_h + ph
            iw = pool_w + pw

            conv_acc = tl.zeros((BLOCK_BATCH, BLOCK_OUTC, BLOCK_H, BLOCK_W), dtype=tl.float32)

            for ic in range(in_c):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        ih_conv = ih + kh - padding
                        iw_conv = iw + kw - padding
                        if ih_conv >= 0 and ih_conv < h and iw_conv >= 0 and iw_conv < w:
                            x_idx = (
                                batch_offs[:, None, None, None] * in_c * h * w +
                                ic * h * w +
                                ih_conv[None, None, :, None] * w +
                                iw_conv[None, None, None, :]
                            )
                            x_val = tl.load(x_ptr + x_idx, mask=batch_mask[:, None, None, None], other=0.0)

                            w_idx = (
                                outc_offs[None, :, None, None] * in_c * kernel_size * kernel_size +
                                ic * kernel_size * kernel_size +
                                kh * kernel_size +
                                kw
                            )
                            w_val = tl.load(w_ptr + w_idx, mask=outc_mask[None, :, None, None], other=0.0)

                            conv_acc += x_val * w_val

            b_idx = outc_offs[None, :, None, None]
            b_val = tl.load(b_ptr + b_idx, mask=outc_mask[None, :, None, None], other=0.0)

            out_val = tl.tanh(conv_acc) * scaling_factor + b_val
            max_val = tl.maximum(max_val, out_val)

    out_idx = (
        batch_offs[:, None, None, None] * out_c * (h // pool_stride) * (w // pool_stride) +
        outc_offs[None, :, None, None] * (h // pool_stride) * (w // pool_stride) +
        oh_offs[None, None, :, None] * (w // pool_stride) +
        ow_offs[None, None, None, :]
    )
    tl.store(out_ptr + out_idx, max_val, mask=batch_mask[:, None, None, None] & outc_mask[None, :, None, None] & oh_mask[None, None, :, None] & ow_mask[None, None, None, :])


def triton_fused_conv_tanh_scale_bias_max(x, w, b, scaling_factor, pool_kernel_size):
    batch, in_c, h, w = x.shape
    out_c, _, kernel_size, _ = w.shape
    stride = 1
    padding = kernel_size // 2
    pool_stride = pool_kernel_size
    out_h = h // pool_stride
    out_w = w // pool_stride

    out = torch.empty((batch, out_c, out_h, out_w), dtype=x.dtype, device=x.device)

    BLOCK_BATCH = 1
    BLOCK_OUTC = 64
    BLOCK_H = 4
    BLOCK_W = 4

    grid = (
        (batch + BLOCK_BATCH - 1) // BLOCK_BATCH,
        (out_c + BLOCK_OUTC - 1) // BLOCK_OUTC,
        (out_h + BLOCK_H - 1) // BLOCK_H,
        (out_w + BLOCK_W - 1) // BLOCK_W,
    )

    fused_conv_tanh_scale_bias_max_kernel[grid](
        x, w, b, out,
        batch, in_c, out_c, h, w,
        kernel_size, stride, padding,
        scaling_factor,
        pool_kernel_size, pool_stride,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUTC=BLOCK_OUTC,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        return triton_fused_conv_tanh_scale_bias_max(x, self.conv_weight, self.bias, self.scaling_factor, self.pool_kernel_size)