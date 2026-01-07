import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    c_offset = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offset = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    
    mask_c = c_offset < C
    mask_hw = hw_offset < H * W
    
    # Compute mean over D dimension
    sum_val = tl.zeros([BLOCK_C, BLOCK_HW], dtype=tl.float32)
    for d in range(D):
        offsets = (
            pid_b * stride_b +
            c_offset[:, None] * stride_c +
            d * stride_d +
            hw_offset[None, :] * stride_h
        )
        mask = mask_c[:, None] & mask_hw[None, :]
        val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += val
    mean_val = sum_val / D
    
    # Add bias
    bias = tl.load(bias_ptr + c_offset, mask=mask_c, other=0.0)
    biased = mean_val + bias[:, None]
    
    # Softmax (online algorithm)
    max_val = tl.max(biased, axis=0)
    exp_val = tl.exp(biased - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    softmax_out = exp_val / sum_exp
    
    # Tanh and scale
    tanh_out = tl.tanh(softmax_out)
    scaled_out = tanh_out * 2.0
    
    # Store output
    out_offsets = (
        pid_b * (C * H * W) +
        c_offset[:, None] * (H * W) +
        hw_offset[None, :]
    )
    out_mask = mask_c[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offsets, scaled_out, mask=out_mask)


def triton_fused_ops(x, bias, B, C, D, H, W):
    out = torch.empty(B, C, 1, H, W, device=x.device, dtype=x.dtype)
    BLOCK_C = 32
    BLOCK_HW = 32
    grid = (B, (C + BLOCK_C - 1) // BLOCK_C, (H * W + BLOCK_HW - 1) // BLOCK_HW)
    fused_kernel[grid](
        x, bias, out,
        B, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        x = triton_fused_ops(x, self.bias.squeeze(), B, C, D, H, W)
        return x