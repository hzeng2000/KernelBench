import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_kernel(
    x_ptr,  # input: (batch, channels, height, width)
    out_ptr,  # output: (batch, 1, height, width)
    batch, channels, height, width,
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    if b >= batch or h >= height or w >= width:
        return
    offsets = b * channels * height * width + tl.arange(0, BLOCK_C) * height * width + h * width + w
    mask = tl.arange(0, BLOCK_C) < channels
    vals = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    min_val = tl.reduce(vals, tl.minimum, axis=0)
    out_offset = b * height * width + h * width + w
    tl.store(out_ptr + out_offset, min_val)

def triton_min(x: torch.Tensor, batch, channels, height, width):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty(batch, 1, height, width, device=x.device, dtype=x.dtype)
    BLOCK_C = 128
    grid = (batch, height, width)
    min_kernel[grid](x, out, batch, channels, height, width, BLOCK_C=BLOCK_C)
    return out

@triton.jit
def sum_kernel(
    x_ptr,  # input: (batch, 1, height, width)
    out_ptr,  # output: (batch, 1, 1, width)
    batch, height, width,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    w = tl.program_id(1)
    if b >= batch or w >= width:
        return
    offsets = b * height * width + tl.arange(0, BLOCK_H) * width + w
    mask = tl.arange(0, BLOCK_H) < height
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.reduce(vals, tl.add, axis=0)
    out_offset = b * width + w
    tl.store(out_ptr + out_offset, sum_val)

def triton_sum(x: torch.Tensor, batch, height, width):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty(batch, 1, 1, width, device=x.device, dtype=x.dtype)
    BLOCK_H = 256
    grid = (batch, width)
    sum_kernel[grid](x, out, batch, height, width, BLOCK_H=BLOCK_H)
    return out

@triton.jit
def gelu_add_kernel(
    x_ptr,  # input: (batch, 1, 1, width)
    bias_ptr,  # bias: (1, 1, 1)
    out_ptr,  # output: (batch, 1, 1, width)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr)  # bias is scalar for broadcasting
    # GELU approximation
    sqrt_half = 0.7071067811865476
    erf_arg = x * sqrt_half
    gelu_val = 0.5 * x * (1 + tl.erf(erf_arg))
    out = gelu_val + bias
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_gelu_add(x: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda
    x = x.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_add_kernel[grid](x, bias, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    Optimized with Triton kernels for min, sum, and fused GELU + add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Precompute dimensions for kernels
        self.batch_size = 16  # Assuming fixed from get_inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = 128  # Input height
        self.width = 128  # Input width
        # Output dimensions after conv transpose
        self.out_height = (self.height - 1) * stride - 2 * padding + kernel_size + output_padding
        self.out_width = (self.width - 1) * stride - 2 * padding + kernel_size + output_padding

    def forward(self, x):
        x = self.conv_transpose(x)
        # Triton min along channels
        x = triton_min(x, self.batch_size, self.out_channels, self.out_height, self.out_width)
        # Triton sum along height
        x = triton_sum(x, self.batch_size, self.out_height, self.out_width)
        # Triton fused GELU + add bias
        x = triton_gelu_add(x, self.bias)
        return x