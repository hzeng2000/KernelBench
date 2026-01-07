import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hardswish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    relu6 = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
    out = x * relu6 / 6.0
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardswish(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    S,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    sum_val = 0.0
    for start in range(0, S, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < S
        vals = tl.load(x_ptr + pid * S + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    tl.store(out_ptr + pid, sum_val / S)


def triton_mean(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    B, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.view(B * C, S).contiguous()
    out = torch.empty(B * C, dtype=x.dtype, device=x.device)
    n = B * C
    BLOCK_SIZE = 128
    grid = (n,)
    mean_kernel[grid](x_flat, out, S, BLOCK_SIZE=BLOCK_SIZE)
    out = out.view(B, C)
    return out


class ModelNew(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish activation
    3. GroupNorm  
    4. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)                             # (B, C, D, H, W)
        x = triton_hardswish(x)                      # Nonlinear activation
        x = self.group_norm(x)                       # Normalization over channels
        x = triton_mean(x)                           # Mean over spatial dims → (B, C)
        return x