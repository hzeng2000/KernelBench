import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reduce_sum_kernel(
    x_ptr, temp_ptr,
    B, C, H, W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    bh = tl.program_id(2)
    bw = tl.program_id(3)
    
    offsets_h = bh * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_w = bw * BLOCK_W + tl.arange(0, BLOCK_W)
    
    mask_h = offsets_h < H
    mask_w = offsets_w < W
    mask = mask_h[:, None] & mask_w[None, :]
    
    x_vals = tl.load(
        x_ptr + b * C * H * W + c * H * W + offsets_h[:, None] * W + offsets_w[None, :],
        mask=mask,
        other=0.0
    )
    sum_val = tl.sum(x_vals)
    
    tl.atomic_add(temp_ptr + b * C + c, sum_val)


@triton.jit
def finalize_avg_bias_kernel(
    temp_ptr, bias_ptr, out_ptr,
    B, C, num_elements,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    
    temp_val = tl.load(temp_ptr + b * C + c)
    avg_val = temp_val / num_elements
    bias_val = tl.load(bias_ptr + c * 1 * 1 + 0 * 1 + 0)
    out_val = avg_val + bias_val
    
    tl.store(out_ptr + b * C * 1 * 1 + c * 1 * 1 + 0 * 1 + 0, out_val)


def triton_global_avg_bias(x: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    
    B, C, H, W = x.shape
    temp = torch.zeros(B, C, device=x.device, dtype=x.dtype)
    out = torch.empty(B, C, 1, 1, device=x.device, dtype=x.dtype)
    
    BLOCK_H = 32
    BLOCK_W = 32
    grid_sum = (
        B,
        C,
        (H + BLOCK_H - 1) // BLOCK_H,
        (W + BLOCK_W - 1) // BLOCK_W,
    )
    reduce_sum_kernel[grid_sum](x, temp, B, C, H, W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W)
    
    grid_final = (B, C)
    finalize_avg_bias_kernel[grid_final](temp, bias, out, B, C, H * W)
    
    return out


@triton.jit
def logsumexp_kernel(
    y_ptr, out_ptr,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    
    offsets = b * C + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < C
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=-float('inf'))
    
    max_y = tl.reduce(tl.max, y_vals)
    sum_exp = tl.reduce(tl.sum, tl.exp(y_vals - max_y))
    logsum = tl.log(sum_exp) + max_y
    
    tl.store(out_ptr + b * 1 * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0, logsum)


def triton_logsumexp(y: torch.Tensor):
    assert y.is_cuda, "Tensor must be on CUDA."
    y = y.contiguous()
    
    B, C, _, _ = y.shape
    out = torch.empty(B, 1, 1, 1, device=y.device, dtype=y.dtype)
    
    BLOCK_SIZE = 128  # Assuming C <= 128
    grid = (B,)
    logsumexp_kernel[grid](y, out, B, C, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_global_avg_bias(x, self.bias)
        x = triton_logsumexp(x)
        x = torch.sum(x, dim=(2, 3))  # Sum over the last two dims (1x1)
        x = x * 10.0  # Multiplication
        return x