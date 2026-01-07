import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def multiply_scalar_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_multiply(x: torch.Tensor, scale: float):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    multiply_scalar_kernel[grid](x, out, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def fused_add_bias_multiply_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    scale2,
    n_elements,
    B, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    total_spatial = D * H * W
    total_per_batch = C * total_spatial
    b = offsets // total_per_batch
    rem = offsets % total_per_batch
    c = rem // total_spatial
    rem2 = rem % total_spatial
    d = rem2 // (H * W)
    rem3 = rem2 % (H * W)
    h = rem3 // W
    w = rem3 % W
    x = tl.load(x_ptr + offsets, mask=mask)
    bias_idx = c
    bias_val = tl.load(bias_ptr + bias_idx, mask=mask)
    out = (x + bias_val) * scale2
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_add_bias_multiply(x: torch.Tensor, bias: torch.Tensor, scale2: float):
    assert x.is_cuda and bias.is_cuda
    x = x.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    B, C, D, H, W = x.shape
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_add_bias_multiply_kernel[grid](x, bias, out, scale2, n_elements, B, C, D, H, W, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_multiply(x, self.scale1.item())
        x = self.avg_pool(x)
        x = triton_fused_add_bias_multiply(x, self.bias, self.scale2.item())
        return x