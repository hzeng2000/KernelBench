import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(
    x_ptr,
    multiplier,
    temp_ptr,
    H,
    W,
    out_channels,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    block_id = tl.program_id(2)
    
    x_offset = b * (out_channels * H * W) + c * (H * W)
    temp_offset = b * (out_channels * num_blocks) + c * num_blocks + block_id
    
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H * W)
    
    x_vals = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0)
    multiplied = x_vals * multiplier
    partial_sum = tl.sum(multiplied, axis=0)
    
    tl.store(temp_ptr + temp_offset, partial_sum)


@triton.jit
def final_reduce_kernel(
    temp_ptr,
    out_ptr,
    H,
    W,
    out_channels,
    num_blocks,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    
    temp_offset = b * (out_channels * num_blocks) + c * num_blocks
    
    offsets = tl.arange(0, num_blocks)
    partials = tl.load(temp_ptr + temp_offset + offsets)
    total_sum = tl.sum(partials, axis=0)
    mean = total_sum / (H * W)
    
    out_offset = b * out_channels + c
    tl.store(out_ptr + out_offset, mean)


def triton_multiply_pool(x: torch.Tensor, multiplier: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch, out_channels, H, W = x.shape
    
    BLOCK_SIZE = 1024
    num_blocks = (H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    temp = torch.empty(batch, out_channels, num_blocks, dtype=torch.float32, device=x.device)
    out = torch.empty(batch, out_channels, 1, 1, dtype=torch.float32, device=x.device)
    
    grid_partial = (batch, out_channels, num_blocks)
    partial_sum_kernel[grid_partial](x, multiplier, temp, H, W, out_channels, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    
    grid_final = (batch, out_channels)
    final_reduce_kernel[grid_final](temp, out, H, W, out_channels, num_blocks)
    
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
    another global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_multiply_pool(x, self.multiplier)
        # The second global average pooling is redundant since after the first it's already 1x1
        return x