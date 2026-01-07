import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_pool_softmax_sub_swish_max_kernel(
    x_ptr, out_ptr, subtract_ptr,
    batch_size, out_channels, d_out, h_out, w_out,
    stride_d, stride_h, stride_w,
    pool_kd, pool_kh, pool_kw,
    pool_sd, pool_sh, pool_sw,
    pool_pd, pool_ph, pool_pw,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D) + pid_d * BLOCK_D
    offs_h = tl.arange(0, BLOCK_H) + pid_h * BLOCK_H
    offs_w = tl.arange(0, BLOCK_W) + pid_w * BLOCK_W

    mask_c = offs_c < out_channels
    mask_d = offs_d < d_out
    mask_h = offs_h < h_out
    mask_w = offs_w < w_out

    # Compute pooled indices
    pool_d_start = tl.maximum(0, (offs_d * stride_d - pool_pd) // pool_sd)
    pool_d_end = tl.minimum(d_out // stride_d, (offs_d * stride_d + pool_kd - pool_pd + pool_sd - 1) // pool_sd)
    pool_h_start = tl.maximum(0, (offs_h * stride_h - pool_ph) // pool_sh)
    pool_h_end = tl.minimum(h_out // stride_h, (offs_h * stride_h + pool_kh - pool_ph + pool_sh - 1) // pool_sh)
    pool_w_start = tl.maximum(0, (offs_w * stride_w - pool_pw) // pool_sw)
    pool_w_end = tl.minimum(w_out // stride_w, (offs_w * stride_w + pool_kw - pool_pw + pool_sw - 1) // pool_sw)

    # Load subtract values
    sub_val = tl.load(subtract_ptr + offs_c, mask=mask_c, other=0.0)

    # Compute softmax across channels
    max_val = float('-inf')
    for c in range(0, out_channels, BLOCK_C):
        offs_c_block = c + tl.arange(0, BLOCK_C)
        mask_c_block = offs_c_block < out_channels
        idx = ((pid_b * out_channels + offs_c_block) * d_out + offs_d[:, None]) * h_out + offs_h[None, :]
        idx = idx * w_out + offs_w[None, None, :]
        x_val = tl.load(x_ptr + idx, mask=mask_c_block[:, None, None] & mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :], other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(x_val))

    exp_sum = 0.0
    for c in range(0, out_channels, BLOCK_C):
        offs_c_block = c + tl.arange(0, BLOCK_C)
        mask_c_block = offs_c_block < out_channels
        idx = ((pid_b * out_channels + offs_c_block) * d_out + offs_d[:, None]) * h_out + offs_h[None, :]
        idx = idx * w_out + offs_w[None, None, :]
        x_val = tl.load(x_ptr + idx, mask=mask_c_block[:, None, None] & mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :], other=0.0)
        exp_val = tl.exp(x_val - max_val)
        exp_sum += tl.sum(exp_val)

    # Apply softmax, subtract, swish, and max
    max_channel = float('-inf')
    for c in range(0, out_channels, BLOCK_C):
        offs_c_block = c + tl.arange(0, BLOCK_C)
        mask_c_block = offs_c_block < out_channels
        idx = ((pid_b * out_channels + offs_c_block) * d_out + offs_d[:, None]) * h_out + offs_h[None, :]
        idx = idx * w_out + offs_w[None, None, :]
        x_val = tl.load(x_ptr + idx, mask=mask_c_block[:, None, None] & mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :], other=0.0)
        softmax_val = tl.exp(x_val - max_val) / exp_sum
        sub_val = softmax_val - sub_val[:, None, None]
        swish_val = tl.sigmoid(sub_val) * sub_val
        max_channel = tl.maximum(max_channel, tl.max(swish_val))

    out_idx = (pid_b * d_out + offs_d) * h_out + offs_h
    out_idx = out_idx * w_out + offs_w
    tl.store(out_ptr + out_idx, max_channel, mask=mask_d & mask_h & mask_w)


def triton_fused_ops(x, subtract, conv_weight, conv_bias, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
    batch_size, in_channels, d_in, h_in, w_in = x.shape
    out_channels = conv_weight.shape[1]
    
    # Compute ConvTranspose3d output size
    d_out = (d_in - 1) * stride[0] - 2 * padding[0] + pool_kernel_size[0] + output_padding[0]
    h_out = (h_in - 1) * stride[1] - 2 * padding[1] + pool_kernel_size[1] + output_padding[1]
    w_out = (w_in - 1) * stride[2] - 2 * padding[2] + pool_kernel_size[2] + output_padding[2]
    
    # Perform ConvTranspose3d using PyTorch (too complex for Triton)
    x = torch.nn.functional.conv_transpose3d(x, conv_weight, conv_bias, stride=stride, padding=padding, output_padding=output_padding)
    
    # Allocate output tensor
    out = torch.empty(batch_size, d_out // pool_stride[0], h_out // pool_stride[1], w_out // pool_stride[2], dtype=x.dtype, device=x.device)
    
    # Launch fused kernel
    BLOCK_C = 16
    BLOCK_D = 4
    BLOCK_H = 8
    BLOCK_W = 8
    
    grid = (batch_size, (d_out + BLOCK_D - 1) // BLOCK_D, (h_out + BLOCK_H - 1) // BLOCK_H, (w_out + BLOCK_W - 1) // BLOCK_W)
    
    fused_transpose_pool_softmax_sub_swish_max_kernel[grid](
        x, out, subtract,
        batch_size, out_channels, d_out, h_out, w_out,
        pool_stride[0], pool_stride[1], pool_stride[2],
        pool_kernel_size[0], pool_kernel_size[1], pool_kernel_size[2],
        pool_stride[0], pool_stride[1], pool_stride[2],
        pool_padding[0], pool_padding[1], pool_padding[2],
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        return triton_fused_ops(x, self.subtract, self.conv_transpose.weight, self.conv_transpose.bias,
                                self.stride, self.padding, self.output_padding,
                                self.pool_kernel_size, self.pool_stride, self.pool_padding)