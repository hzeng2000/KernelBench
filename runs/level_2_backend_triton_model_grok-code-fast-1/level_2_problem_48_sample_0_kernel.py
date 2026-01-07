import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,  # Pointer to conv output
    scaling_ptr,  # Pointer to scaling_factor (shape: C)
    bias_ptr,  # Pointer to bias (shape: C)
    out_ptr,  # Pointer to output
    B, C, D, H, W,  # Dimensions
    n_elements,  # Total elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute indices from flat offset
    total_per_batch = C * D * H * W
    b = offsets // total_per_batch
    rem = offsets % total_per_batch
    total_per_channel = D * H * W
    c = rem // total_per_channel
    rem2 = rem % total_per_channel
    d = rem2 // (H * W)
    h = (rem2 % (H * W)) // W
    w = rem2 % W

    # Load x
    x = tl.load(x_ptr + offsets, mask=mask)
    # Load scaling and bias (broadcasted over channels)
    scaling = tl.load(scaling_ptr + c, mask=mask)
    bias = tl.load(bias_ptr + c, mask=mask)

    # Compute: sigmoid(scaling * tanh(x) * bias)
    out = scaling * tl.tanh(x) * bias
    out = 1.0 / (1.0 + tl.exp(-out))  # sigmoid

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_post_conv(x: torch.Tensor, scaling: torch.Tensor, bias: torch.Tensor):
    """
    Fused kernel for post-conv operations: sigmoid(scaling * tanh(x) * bias)
    """
    assert x.is_cuda and scaling.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    scaling = scaling.contiguous()
    bias = bias.contiguous()

    out = torch.empty_like(x)
    B, C, D, H, W = x.shape
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_post_conv_kernel[grid](
        x, scaling, bias, out, B, C, D, H, W, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, then fuses the post-conv element-wise operations into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = fused_post_conv(x, self.scaling_factor, self.bias)
        return x