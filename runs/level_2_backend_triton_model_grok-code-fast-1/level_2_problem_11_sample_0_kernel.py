import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_tanh_maxpool_kernel(
    x_ptr,
    y_ptr,
    batch_size,
    out_channels,
    H, W,
    H_out, W_out,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_CH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    batch_start = pid_batch * BLOCK_BATCH
    ch_start = pid_ch * BLOCK_CH
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    batch_offsets = batch_start + tl.arange(0, BLOCK_BATCH)
    ch_offsets = ch_start + tl.arange(0, BLOCK_CH)
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)

    mask_batch = batch_offsets < batch_size
    mask_ch = ch_offsets < out_channels
    mask_h = h_offsets < H_out
    mask_w = w_offsets < W_out
    mask = mask_batch[:, None, None, None] & mask_ch[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :]

    h_out = h_offsets[None, None, :, None]
    w_out = w_offsets[None, None, None, :]
    h0 = 2 * h_out
    h1 = 2 * h_out + 1
    w0 = 2 * w_out
    w1 = 2 * w_out + 1

    mask_load0 = mask & (h0 < H) & (w0 < W)
    x0 = tl.load(x_ptr + batch_offsets[:, None, None, None] * (out_channels * H * W) + ch_offsets[None, :, None, None] * (H * W) + h0 * W + w0, mask=mask_load0, other=-float('inf'))
    mask_load1 = mask & (h0 < H) & (w1 < W)
    x1 = tl.load(x_ptr + batch_offsets[:, None, None, None] * (out_channels * H * W) + ch_offsets[None, :, None, None] * (H * W) + h0 * W + w1, mask=mask_load1, other=-float('inf'))
    mask_load2 = mask & (h1 < H) & (w0 < W)
    x2 = tl.load(x_ptr + batch_offsets[:, None, None, None] * (out_channels * H * W) + ch_offsets[None, :, None, None] * (H * W) + h1 * W + w0, mask=mask_load2, other=-float('inf'))
    mask_load3 = mask & (h1 < H) & (w1 < W)
    x3 = tl.load(x_ptr + batch_offsets[:, None, None, None] * (out_channels * H * W) + ch_offsets[None, :, None, None] * (H * W) + h1 * W + w1, mask=mask_load3, other=-float('inf'))

    tanh_x0 = tl.tanh(x0)
    tanh_x1 = tl.tanh(x1)
    tanh_x2 = tl.tanh(x2)
    tanh_x3 = tl.tanh(x3)

    max_val = tl.maximum(tl.maximum(tanh_x0, tanh_x1), tl.maximum(tanh_x2, tanh_x3))

    y_ptr_offset = batch_offsets[:, None, None, None] * (out_channels * H_out * W_out) + ch_offsets[None, :, None, None] * (H_out * W_out) + h_offsets[None, None, :, None] * W_out + w_offsets[None, None, None, :]
    tl.store(y_ptr + y_ptr_offset, max_val, mask=mask)

def triton_fused_tanh_maxpool(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    batch_size, out_channels, H, W = x.shape
    H_out = H // 2
    W_out = W // 2
    y = torch.empty(batch_size, out_channels, H_out, W_out, dtype=x.dtype, device=x.device)
    BLOCK_BATCH = 1
    BLOCK_CH = 32
    BLOCK_H = 8
    BLOCK_W = 8
    grid = ((batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH, (out_channels + BLOCK_CH - 1) // BLOCK_CH, (H_out + BLOCK_H - 1) // BLOCK_H, (W_out + BLOCK_W - 1) // BLOCK_W)
    fused_tanh_maxpool_kernel[grid](x, y, batch_size, out_channels, H, W, H_out, W_out, BLOCK_BATCH=BLOCK_BATCH, BLOCK_CH=BLOCK_CH, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W)
    return y

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, fused tanh + max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = triton_fused_tanh_maxpool(x)
        x = self.group_norm(x)
        return x