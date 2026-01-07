import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_min_tanh_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, h, w, k,
    stride_h, stride_w, pad_h, pad_w,
    out_h, out_w,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    batch_offs = pid_b * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    out_c_offs = pid_oc * BLOCK_OUT_C + tl.arange(0, BLOCK_OUT_C)
    oh_offs = pid_oh * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_offs = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)

    batch_mask = batch_offs < batch
    out_c_mask = out_c_offs < out_c
    oh_mask = oh_offs < out_h
    ow_mask = ow_offs < out_w

    # Compute min across channels for each spatial position
    min_vals = tl.full([BLOCK_BATCH, BLOCK_H, BLOCK_W], float('inf'), dtype=tl.float32)
    
    for ic in range(in_c):
        for kh in range(k):
            for kw in range(k):
                ih = oh_offs * stride_h - pad_h + kh
                iw = ow_offs * stride_w - pad_w + kw
                
                h_mask = (ih >= 0) & (ih < h)
                w_mask = (iw >= 0) & (iw < w)
                valid_mask = batch_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
                
                x_idx = (
                    batch_offs[:, None, None] * in_c * h * w +
                    ic * h * w +
                    ih[None, :, None] * w +
                    iw[None, None, :]
                )
                x_val = tl.load(x_ptr + x_idx, mask=valid_mask, other=0.0)
                
                w_idx = (
                    out_c_offs[:, None, None] * in_c * k * k +
                    ic * k * k +
                    kh * k +
                    kw
                )
                w_val = tl.load(w_ptr + w_idx, mask=out_c_mask[:, None, None], other=0.0)
                
                acc = tl.sum(x_val[None, :, :] * w_val[:, None, None], axis=0)
                
                if ic == 0 and kh == 0 and kw == 0:
                    min_vals = acc
                else:
                    min_vals = tl.minimum(min_vals, acc)
    
    # Add bias
    b_val = tl.load(b_ptr + out_c_offs, mask=out_c_mask, other=0.0)
    min_vals = min_vals + b_val[:, None, None]
    
    # Apply tanh twice
    min_vals = tl.tanh(min_vals)
    min_vals = tl.tanh(min_vals)
    
    # Store result
    out_idx = (
        batch_offs[:, None, None] * out_h * out_w +
        oh_offs[None, :, None] * out_w +
        ow_offs[None, None, :]
    )
    tl.store(out_ptr + out_idx, min_vals, mask=batch_mask[:, None, None] & oh_mask[None, :, None] & ow_mask[None, None, :])


def triton_fused_conv_min_tanh_tanh(x, weight, bias, stride=1, padding=0):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    batch, in_c, h, w = x.shape
    out_c, _, k, _ = weight.shape
    
    stride_h = stride_w = stride
    pad_h = pad_w = padding
    
    out_h = (h + 2 * pad_h - k) // stride_h + 1
    out_w = (w + 2 * pad_w - k) // stride_w + 1
    
    out = torch.empty(batch, 1, out_h, out_w, dtype=x.dtype, device=x.device)
    
    BLOCK_BATCH = 4
    BLOCK_OUT_C = 64
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid = (
        (batch + BLOCK_BATCH - 1) // BLOCK_BATCH,
        (out_c + BLOCK_OUT_C - 1) // BLOCK_OUT_C,
        (out_h + BLOCK_H - 1) // BLOCK_H,
        (out_w + BLOCK_W - 1) // BLOCK_W,
    )
    
    fused_conv_min_tanh_tanh_kernel[grid](
        x, weight, bias, out,
        batch, in_c, out_c, h, w, k,
        stride_h, stride_w, pad_h, pad_w,
        out_h, out_w,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_C=BLOCK_OUT_C,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return triton_fused_conv_min_tanh_tanh(x, weight, bias, stride=1, padding=0)