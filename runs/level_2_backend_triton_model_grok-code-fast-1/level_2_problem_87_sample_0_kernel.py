import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mish_subtract_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    val,  # Scalar to subtract
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
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Subtract val
    x_sub = x - val
    # Compute mish: x_sub * tanh(softplus(x_sub))
    softplus = tl.log(1.0 + tl.exp(x_sub))
    tanh_softplus = tl.tanh(softplus)
    out = x_sub * tanh_softplus
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish_subtract(x: torch.Tensor, val: float):
    """
    This function wraps the Triton kernel call for fused subtract and mish.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    mish_subtract_kernel[grid](x, out, val, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, then fused subtract and Mish activation using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        x = self.conv(x)
        # Fused subtract (subtract_value_1 + subtract_value_2) and mish using Triton
        val = self.subtract_value_1 + self.subtract_value_2
        return triton_mish_subtract(x, val)