import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bias_tanh_kernel(
    x_ptr,  # Pointer to input tensor (output of conv_transpose)
    bias_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    batch_size,
    out_channels,
    height,
    width,
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute indices for broadcasting bias
    b = offsets // (out_channels * height * width)
    rem = offsets % (out_channels * height * width)
    c = rem // (height * width)
    # h and w not needed for bias, but computed for completeness
    # rem2 = rem % (height * width)
    # h = rem2 // width
    # w = rem2 % width

    # Load x value
    x_val = tl.load(x_ptr + offsets, mask=mask)
    # Load bias value (broadcasted per channel)
    bias_val = tl.load(bias_ptr + c, mask=mask)
    # Compute x - bias, then tanh
    out_val = tl.tanh(x_val - bias_val)
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)


def triton_bias_tanh(x: torch.Tensor, bias: torch.Tensor):
    """
    Fused bias subtraction and tanh using Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()

    batch_size, out_channels, height, width = x.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    bias_tanh_kernel[grid](
        x, bias, out, batch_size, out_channels, height, width, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused bias subtraction and tanh using Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_bias_tanh(x, self.bias)
        return x