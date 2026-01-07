import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_conv_kernel(
    x_ptr,  # Pointer to conv output
    sum_ptr,  # Pointer to sum_tensor
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    B, C, D, H, W,  # Shapes
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index: c = (offsets // (D * H * W)) % C
    chw = D * H * W
    c = (offsets // chw) % C

    # Load sum_val for this channel
    sum_val = tl.load(sum_ptr + c, mask=mask, other=0.0)

    # LeakyReLU: x if x > 0 else 0.2 * x
    leaky = tl.where(x > 0, x, 0.2 * x)

    # Add sum_tensor
    added = leaky + sum_val

    # Clamp
    clamped = tl.maximum(tl.minimum(added, 1.0), -1.0)

    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = tl.sqrt(2.0 / math.pi)
    x3 = clamped * clamped * clamped
    tanh_arg = sqrt_2_pi * (clamped + 0.044715 * x3)
    gelu = 0.5 * clamped * (1.0 + tl.tanh(tanh_arg))

    # Store
    tl.store(out_ptr + offsets, gelu, mask=mask)


def fused_post_conv(x: torch.Tensor, sum_tensor: torch.Tensor):
    assert x.is_cuda and sum_tensor.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    sum_tensor = sum_tensor.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    B, C, D, H, W = x.shape

    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_post_conv_kernel[grid](
        x, sum_tensor, out, n_elements, B, C, D, H, W, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_post_conv(x, self.sum_tensor)
        return x