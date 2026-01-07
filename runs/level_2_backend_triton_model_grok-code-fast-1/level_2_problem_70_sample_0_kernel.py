import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_sigmoid_scale_residual_kernel(
    x_ptr,  # Pointer to input/output (in-place modification or new output)
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
    # Compute sigmoid
    sig = tl.sigmoid(x)
    # Scale and add residual
    out = sig * scaling_factor + x
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_sigmoid_scale_residual(x: torch.Tensor, scaling_factor: float):
    """
    Fused kernel for sigmoid, scaling, and residual addition.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable, chosen for good occupancy
    # Grid
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch kernel
    fused_sigmoid_scale_residual_kernel[grid](x, out, scaling_factor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd" with Triton kernel for post-GEMM operations.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = self.gemm(x)
        x = triton_fused_sigmoid_scale_residual(x, self.scaling_factor)
        return x