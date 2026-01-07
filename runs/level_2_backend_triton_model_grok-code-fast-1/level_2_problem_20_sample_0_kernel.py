import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,  # Pointer to x_conv
    bias_ptr,  # Pointer to bias (flattened to (C,))
    out_ptr,  # Pointer to output
    B, C, D, H, W,  # Shapes
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute indices
    total_per_batch = C * D * H * W
    b = offsets // total_per_batch
    rem = offsets % total_per_batch
    total_per_channel = D * H * W
    c = rem // total_per_channel
    rem2 = rem % total_per_channel
    total_per_depth = H * W
    d = rem2 // total_per_depth
    rem3 = rem2 % total_per_depth
    h = rem3 // W
    w = rem3 % W

    # Load x_conv
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load bias (broadcasted per channel)
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # Compute: ((x + bias) + x) * x + x
    temp = x + bias_val
    temp = temp + x
    temp = temp * x
    out_val = temp + x

    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)


def fused_post_conv(x_conv: torch.Tensor, bias: torch.Tensor):
    """
    Fused kernel for post-convolution operations.
    """
    assert x_conv.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x_conv = x_conv.contiguous()
    bias = bias.contiguous().view(-1)  # Flatten bias to (C,)

    B, C, D, H, W = x_conv.shape
    out = torch.empty_like(x_conv)
    n_elements = x_conv.numel()
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_post_conv_kernel[grid](
        x_conv, bias, out, B, C, D, H, W, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model with fused Triton kernel for post-convolution operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use fused Triton kernel for the remaining operations
        x = fused_post_conv(x, self.bias)
        return x