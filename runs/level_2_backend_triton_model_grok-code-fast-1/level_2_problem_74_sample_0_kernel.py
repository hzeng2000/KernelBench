import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_leaky_mul_leaky_kernel(
    x_ptr,  # Pointer to input tensor (after conv)
    mul_ptr,  # Pointer to multiplier tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    channels,  # Number of channels
    depth,  # Depth dimension
    height,  # Height dimension
    width,  # Width dimension
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for broadcasting multiplier
    stride_dhw = depth * height * width
    stride_c = channels * stride_dhw
    channel_idx = (offsets % stride_c) // stride_dhw

    # Load multiplier (broadcasted)
    mul = tl.load(mul_ptr + channel_idx, mask=mask, other=0.0)

    # First LeakyReLU: max(x, 0.2 * x)
    leaky1 = tl.where(x > 0, x, 0.2 * x)

    # Multiply by multiplier
    after_mul = leaky1 * mul

    # Second LeakyReLU: max(after_mul, 0.2 * after_mul)
    out = tl.where(after_mul > 0, after_mul, 0.2 * after_mul)

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_leaky_mul_leaky(x: torch.Tensor, multiplier: torch.Tensor):
    assert x.is_cuda and multiplier.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    multiplier = multiplier.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Get shapes
    batch, channels, depth, height, width = x.shape
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size

    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    fused_leaky_mul_leaky_kernel[grid](
        x, multiplier, out, n_elements, channels, depth, height, width, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused LeakyReLU, multiplication, and LeakyReLU replaced by a custom Triton kernel.
    ConvTranspose3d and MaxPool3d remain unchanged.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_fused_leaky_mul_leaky(x, self.multiplier)
        x = self.max_pool(x)
        return x