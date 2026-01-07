import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_scale_pool_bias_scale_kernel(
    x_ptr, out_ptr,
    scale1, scale2,
    bias_ptr,
    B, C_out, D_out, H_out, W_out,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    pool_d, pool_h, pool_w,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_hw = tl.program_id(3)

    hw_start = pid_hw * BLOCK_H * BLOCK_W
    d_start = pid_d * BLOCK_D

    offs_d = d_start + tl.arange(0, BLOCK_D)
    offs_h = (hw_start // W_out) + tl.arange(0, BLOCK_H)
    offs_w = (hw_start % W_out) + tl.arange(0, BLOCK_W)

    mask_d = offs_d < (D_out // 2)
    mask_h = offs_h < (H_out // 2)
    mask_w = offs_w < (W_out // 2)

    pool_size = pool_d * pool_h * pool_w

    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

    for pd in range(pool_d):
        for ph in range(pool_h):
            for pw in range(pool_w):
                idx_d = offs_d * pool_d + pd
                idx_h = offs_h * pool_h + ph
                idx_w = offs_w * pool_w + pw

                mask_full = (idx_d < D_out) & (idx_h < H_out) & (idx_w < W_out)
                idx = (pid_b * stride_b +
                       pid_c * stride_c +
                       idx_d * stride_d +
                       idx_h * stride_h +
                       idx_w * stride_w)

                val = tl.load(x_ptr + idx, mask=mask_full, other=0.0)
                acc += val

    out_val = acc / pool_size
    out_val = out_val * scale1
    bias = tl.load(bias_ptr + pid_c)
    out_val = out_val + bias
    out_val = out_val * scale2

    out_idx = (pid_b * (C_out * (D_out//2) * (H_out//2) * (W_out//2)) +
               pid_c * ((D_out//2) * (H_out//2) * (W_out//2)) +
               (offs_d[:, None, None] * (H_out//2) * (W_out//2) +
                offs_h[None, :, None] * (W_out//2) +
                offs_w[None, None, :]))

    mask_out = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    tl.store(out_ptr + out_idx, out_val, mask=mask_out)


def triton_conv_transpose_scale_pool_bias_scale(x, weight, bias_conv, scale1, scale2, bias, stride, padding):
    B, C_in, D_in, H_in, W_in = x.shape
    C_out, _, kD, kH, kW = weight.shape
    stride_d, stride_h, stride_w = stride, stride, stride
    pad_d, pad_h, pad_w = padding, padding, padding

    D_out = (D_in - 1) * stride_d - 2 * pad_d + kD
    H_out = (H_in - 1) * stride_h - 2 * pad_h + kH
    W_out = (W_in - 1) * stride_w - 2 * pad_w + kW

    x_unfold = torch.nn.functional.conv_transpose3d(x, weight, bias_conv, stride=stride, padding=padding)
    out = torch.empty(B, C_out, D_out//2, H_out//2, W_out//2, device=x.device, dtype=x.dtype)

    BLOCK_C = 4
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4

    grid = (B, C_out, (D_out//2 + BLOCK_D - 1) // BLOCK_D, ((H_out//2)*(W_out//2) + BLOCK_H*BLOCK_W - 1) // (BLOCK_H*BLOCK_W))

    fused_transpose_scale_pool_bias_scale_kernel[grid](
        x_unfold, out,
        scale1.item(), scale2.item(),
        bias,
        B, C_out, D_out, H_out, W_out,
        x_unfold.stride(0), x_unfold.stride(1), x_unfold.stride(2), x_unfold.stride(3), x_unfold.stride(4),
        2, 2, 2,
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        return triton_conv_transpose_scale_pool_bias_scale(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.scale1, self.scale2, self.bias,
            stride=2, padding=1
        )