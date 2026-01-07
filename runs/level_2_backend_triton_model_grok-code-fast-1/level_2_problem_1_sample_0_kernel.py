import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_bias_kernel(
    x_ptr,  # Pointer to input tensor (output of conv)
    bias_ptr,  # Pointer to bias tensor (shape: out_channels)
    out_ptr,  # Pointer to output tensor
    total_elements,  # Total number of elements in the tensor
    out_channels,  # Number of output channels
    height,  # Height of the feature map
    width,  # Width of the feature map
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute indices
    channel_stride = height * width
    batch_channel_stride = out_channels * channel_stride
    batch_idx = offsets // batch_channel_stride
    channel_idx = (offsets // channel_stride) % out_channels
    h_idx = (offsets // width) % height
    w_idx = offsets % width

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load bias (broadcasted per channel)
    bias_val = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    # Apply ReLU and add bias
    out = tl.maximum(x, 0.0) + bias_val
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu_bias(x: torch.Tensor, bias: torch.Tensor, out_channels: int, height: int, width: int):
    """
    Wrapper for the Triton kernel that applies ReLU and adds bias.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Total elements
    total_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size

    # Grid
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    relu_bias_kernel[grid](
        x, bias, out, total_elements, out_channels, height, width, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies ReLU, and adds a bias term using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        # Get height and width from the tensor shape
        height, width = x.shape[2], x.shape[3]
        # Apply ReLU and add bias using Triton kernel
        x = triton_relu_bias(x, self.bias, self.out_channels, height, width)
        return x