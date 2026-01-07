import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def final_kernel(
    x_ptr,  # Pointer to input tensor (batch, out_channels, 1, 1, 1)
    bias_ptr,  # Pointer to bias tensor (out_channels, 1, 1, 1)
    out_ptr,  # Pointer to output tensor (batch,)
    divisor,  # Scalar divisor
    batch,  # Batch size
    out_channels,  # Number of out_channels
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_channels
    x_vals = tl.load(x_ptr + b * out_channels + offsets, mask=mask)
    bias_vals = tl.load(bias_ptr + offsets, mask=mask)
    vals = (x_vals / divisor) + bias_vals
    sum_val = tl.sum(vals)
    tl.store(out_ptr + b, sum_val)


def triton_final(x: torch.Tensor, bias: torch.Tensor, divisor: float, sum_dim: int):
    assert sum_dim == 1, "Only sum_dim=1 is supported"
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    batch, out_channels = x.shape[0], x.shape[1]
    out = torch.empty(batch, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = out_channels
    grid = (batch,)
    final_kernel[grid](x, bias, out, divisor, batch, out_channels, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, global average pooling,
    divides by a constant, adds a bias term, and sums along a specific dimension.
    The division, bias addition, and summation are fused into a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        # Fuse division, bias addition, and sum using Triton
        return triton_final(x, self.bias, self.divisor, self.sum_dim)