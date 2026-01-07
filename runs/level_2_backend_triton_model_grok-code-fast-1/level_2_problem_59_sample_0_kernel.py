import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_scale_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    scaling_factor,  # Scaling factor
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
    # Compute Swish: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    swish = x * sigmoid_x
    # Apply scaling
    out = swish * scaling_factor
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish_scale(x: torch.Tensor, scaling_factor: float):
    """
    This function wraps the Triton kernel call for Swish activation followed by scaling.
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
    swish_scale_kernel[grid](x, out, scaling_factor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies Swish activation, and scales the result.
    The Swish activation and scaling are fused into a single Triton kernel for efficiency.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        x = triton_swish_scale(x, self.scaling_factor)
        return x