import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def post_conv_kernel(
    conv_out_ptr,  # Pointer to conv output
    bias_ptr,      # Pointer to bias (shape: out_channels, 1, 1)
    out_ptr,       # Pointer to output
    constant,      # Scalar constant for min
    scale,         # Scalar scaling factor
    batch_size,
    out_channels,
    height,
    width,
    num_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Compute indices
    b = offsets // (out_channels * height * width)
    c = (offsets // (height * width)) % out_channels
    h = (offsets // width) % height
    w = offsets % width

    # Load conv output
    conv_val = tl.load(conv_out_ptr + offsets, mask=mask, other=0.0)
    # Load bias (bias is (out_channels, 1, 1), so index by c)
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # Compute: (min(conv_val, constant) + bias_val) * scale
    min_val = tl.minimum(conv_val, constant)
    result = (min_val + bias_val) * scale

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_post_conv(conv_out: torch.Tensor, bias: torch.Tensor, constant: float, scale: float):
    """
    Fused kernel for min, add bias, and mul scale after conv.
    """
    assert conv_out.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    conv_out = conv_out.contiguous()
    bias = bias.contiguous()

    # Output tensor
    out = torch.empty_like(conv_out)

    batch_size, out_channels, height, width = conv_out.shape
    num_elements = conv_out.numel()
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    post_conv_kernel[grid](
        conv_out, bias, out, constant, scale,
        batch_size, out_channels, height, width, num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused post-conv operations using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        # Fused min, add bias, mul scale
        x = triton_post_conv(x, self.bias, self.constant_value, self.scaling_factor)
        return x