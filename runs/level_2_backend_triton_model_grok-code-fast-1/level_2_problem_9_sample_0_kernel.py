import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel(
    x_ptr,  # Pointer to input tensor (output of linear)
    out_ptr,  # Pointer to output tensor
    subtract_value,  # Scalar to subtract
    multiply_value,  # Scalar to multiply
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute (x - subtract_value) * multiply_value, then relu
    out = tl.maximum((x - subtract_value) * multiply_value, 0.0)
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_post(x: torch.Tensor, subtract_value: float, multiply_value: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_post_kernel[grid](x, out, subtract_value, multiply_value, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, fused subtraction, multiplication, and ReLU activation using Triton.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        # Fused operation: subtract, multiply, relu
        x = triton_fused_post(x, self.subtract_value, self.multiply_value)
        return x