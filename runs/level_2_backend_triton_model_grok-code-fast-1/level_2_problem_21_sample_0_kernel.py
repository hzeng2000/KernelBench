import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bias_scale_sigmoid_kernel(
    x_ptr,  # Pointer to input tensor x (N, C, H, W)
    bias_ptr,  # Pointer to bias (C, 1, 1), but treated as (C,) in memory
    scale_ptr,  # Pointer to scale (C, 1, 1), but treated as (C,) in memory
    out_ptr,  # Pointer to output tensor
    total_elements,  # Total number of elements
    N, C, H, W,  # Shapes
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute indices from flat offset
    offset = offsets
    w = offset % W
    offset = offset // W
    h = offset % H
    offset = offset // H
    c = offset % C
    n = offset // C

    # Load x
    x = tl.load(x_ptr + offsets, mask=mask)
    # Load bias and scale using channel index c
    bias = tl.load(bias_ptr + c, mask=mask)
    scale = tl.load(scale_ptr + c, mask=mask)

    # Compute fused operation: sigmoid((x + bias) * scale)
    out = tl.sigmoid((x + bias) * scale)
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_bias_scale_sigmoid(x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor):
    assert x.is_cuda and bias.is_cuda and scale.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    scale = scale.contiguous()

    N, C, H, W = x.shape
    out = torch.empty_like(x)
    total_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size

    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_bias_scale_sigmoid_kernel[grid](
        x, bias, scale, out, total_elements, N, C, H, W, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    Optimized with a fused Triton kernel for bias addition, scaling, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = triton_fused_bias_scale_sigmoid(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x