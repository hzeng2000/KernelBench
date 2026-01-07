import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_hardswish_kernel(
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
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute ReLU
    relu_x = tl.maximum(0.0, x)
    # Compute HardSwish: relu_x * clamp((relu_x + 3) / 6, 0, 1)
    clamp_val = tl.clamp((relu_x + 3.0) / 6.0, 0.0, 1.0)
    out = relu_x * clamp_val
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu_hardswish(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for fused ReLU + HardSwish.
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
    relu_hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution and applies fused ReLU + HardSwish using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_relu_hardswish(x)
        return x