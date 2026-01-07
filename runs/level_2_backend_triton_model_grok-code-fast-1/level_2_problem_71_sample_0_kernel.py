import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_div_leaky_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    divisor,  # Division constant
    negative_slope,  # Negative slope for LeakyReLU
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    # Perform division
    out = x / divisor
    # Apply LeakyReLU: out = max(out, negative_slope * out) but actually LeakyReLU is out if out >= 0 else negative_slope * out
    out = tl.where(out >= 0, out, negative_slope * out)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_div_leaky(x: torch.Tensor, divisor: float, negative_slope: float):
    """
    Fused division and LeakyReLU using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch kernel
    fused_div_leaky_kernel[grid](x, out, divisor, negative_slope, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused division and LeakyReLU using Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.conv(x)
        x = triton_fused_div_leaky(x, self.divisor, 0.01)
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]