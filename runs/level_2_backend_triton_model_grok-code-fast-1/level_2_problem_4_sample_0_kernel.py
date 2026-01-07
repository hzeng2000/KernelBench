import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mish_twice_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
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
    # Compute mish(x) = x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(x))
    t = tl.tanh(sp)
    mish_out = x * t
    # Compute mish again on the result
    sp2 = tl.log(1.0 + tl.exp(mish_out))
    t2 = tl.tanh(sp2)
    out = mish_out * t2
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish_twice(x: torch.Tensor):
    """
    Applies mish twice using Triton kernel.
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
    mish_twice_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model with Triton kernel for mish applied twice.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_mish_twice(x)
        return x