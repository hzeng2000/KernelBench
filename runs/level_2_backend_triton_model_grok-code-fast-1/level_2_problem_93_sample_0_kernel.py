import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,  # Pointer to input tensor (output of conv)
    out_ptr,  # Pointer to output tensor
    add_val,  # Scalar to add
    mul_val,  # Scalar to multiply
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
    # Add scalar
    x = x + add_val
    # ReLU (min with 0)
    x = tl.minimum(x, 0.0)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = tl.sqrt(2.0 / 3.141592653589793)
    x_cubed = x * x * x
    inner = sqrt_2_pi * (x + 0.044715 * x_cubed)
    tanh_inner = tl.tanh(inner)
    x = 0.5 * x * (1.0 + tanh_inner)
    # Multiply by scalar
    x = x * mul_val
    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_post_conv(x: torch.Tensor, add_val: float, mul_val: float):
    """
    Fused kernel for add scalar, ReLU, GELU, multiply scalar.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    # Prepare output tensor
    out = torch.empty_like(x)
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    # Grid
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch kernel
    fused_post_conv_kernel[grid](x, out, add_val, mul_val, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model with fused Triton kernel for post-conv operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_conv(x, self.add_value, self.multiply_value)
        return x