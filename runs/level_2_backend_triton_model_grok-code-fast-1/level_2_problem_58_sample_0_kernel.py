import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(
    x_ptr,  # Input: (B, C, D, H, W)
    out_ptr,  # Output: (B, 1, D, H, W)
    B, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,  # Not really used, since C is small
):
    # Each program handles one (b, d, h, w)
    pid = tl.program_id(0)
    total_spatial = D * H * W
    b = pid // total_spatial
    spatial_idx = pid % total_spatial
    d = spatial_idx // (H * W)
    hw = spatial_idx % (H * W)
    h = hw // W
    w = hw % W
    
    # Offsets for the C dimension
    offsets = tl.arange(0, C)
    x_offsets = b * (C * D * H * W) + offsets * (D * H * W) + d * (H * W) + h * W + w
    x = tl.load(x_ptr + x_offsets)
    
    # Compute max
    max_x = tl.max(x)
    # Compute sum exp(x - max_x)
    exp_x = tl.exp(x - max_x)
    sum_exp = tl.sum(exp_x)
    # logsumexp
    lse = tl.log(sum_exp) + max_x
    
    # Store to out
    out_offset = b * (D * H * W) + d * (H * W) + h * W + w
    tl.store(out_ptr + out_offset, lse)


def triton_logsumexp(x: torch.Tensor):
    assert x.is_cuda and x.dim() == 5
    B, C, D, H, W = x.shape
    out = torch.empty(B, 1, D, H, W, dtype=x.dtype, device=x.device)
    
    grid = (B * D * H * W,)
    logsumexp_kernel[grid](x, out, B, C, D, H, W, BLOCK_SIZE=1)  # BLOCK_SIZE not used
    return out


@triton.jit
def hardswish_sub_clamp_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr)  # scalar
    
    # HardSwish: x * sigmoid(x + 3) / 6
    sig = 1.0 / (1.0 + tl.exp(-(x + 3.0)))
    hardswish = x * sig / 6.0
    
    # Subtract bias
    result = hardswish - bias
    
    # Clamp to [-1, 1]
    result = tl.maximum(tl.minimum(result, 1.0), -1.0)
    
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_hardswish_sub_clamp(x: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    hardswish_sub_clamp_kernel[grid](x, bias, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1)) 

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_logsumexp(x)
        x = triton_hardswish_sub_clamp(x, self.bias)
        return x