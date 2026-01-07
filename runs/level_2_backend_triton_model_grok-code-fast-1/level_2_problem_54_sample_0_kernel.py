import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel(
    x_ptr,  # Pointer to input (after conv)
    mult_ptr,  # Pointer to multiplier (1D, size out_channels)
    out_ptr,  # Pointer to output
    batch_size,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    n_elements = batch_size * out_channels * height * width
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute channel index
    hw = height * width
    chw = out_channels * hw
    temp = offsets % chw
    channel = temp // hw
    # Load multiplier
    mult = tl.load(mult_ptr + channel, mask=mask, other=0.0)
    # Multiply
    y = x * mult
    # LeakyReLU
    z = tl.where(y > 0, y, 0.01 * y)
    # GELU
    tanh_arg = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    tanh_val = tl.tanh(tanh_arg)
    w = 0.5 * z * (1 + tanh_val)
    # Store
    tl.store(out_ptr + offsets, w, mask=mask)


def fused_post(x: torch.Tensor, mult: torch.Tensor, batch_size, out_channels, height, width):
    assert x.is_cuda and mult.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    mult = mult.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_post_kernel[grid](x, mult, out, batch_size, out_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        batch_size, out_channels, height, width = x.shape
        mult_flat = self.multiplier.view(-1)
        x = fused_post(x, mult_flat, batch_size, out_channels, height, width)
        return x