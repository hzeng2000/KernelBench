import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_norm_ptr,  # Pointer to x_norm
    x_conv_ptr,  # Pointer to x_conv
    out_ptr,     # Pointer to output x_res
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_norm = tl.load(x_norm_ptr + offsets, mask=mask, other=0.0)
    x_conv = tl.load(x_conv_ptr + offsets, mask=mask, other=0.0)
    # Compute tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    exp_x = tl.exp(x_norm)
    exp_neg_x = tl.exp(-x_norm)
    tanh_x = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    # Compute hardswish: x * clamp(x + 3, 0, 6) / 6
    hardswish_in = tanh_x + 3.0
    clamped = tl.clamp(hardswish_in, 0.0, 6.0)
    hardswish_x = tanh_x * clamped / 6.0
    # Residual addition
    out = x_conv + hardswish_x
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused(x_norm: torch.Tensor, x_conv: torch.Tensor):
    assert x_norm.is_cuda and x_conv.is_cuda, "Tensors must be on CUDA."
    assert x_norm.shape == x_conv.shape, "Shapes must match."
    x_norm = x_norm.contiguous()
    x_conv = x_conv.contiguous()
    out = torch.empty_like(x_norm)
    n_elements = x_norm.numel()
    BLOCK_SIZE = 1024  # Tunable
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_kernel[grid](x_norm, x_conv, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def logsumexp_kernel(
    x_ptr,      # Pointer to x_res
    out_ptr,    # Pointer to output
    batch, channels, h, w,
):
    pid = tl.program_id(0)
    b = pid // (h * w)
    hw = pid % (h * w)
    hi = hw // w
    wi = hw % w
    # Offsets for the channel dimension
    offsets = b * (channels * h * w) + tl.arange(0, channels) * (h * w) + hi * w + wi
    x = tl.load(x_ptr + offsets)
    # Compute max
    max_val = tl.max(x)
    # Compute sum of exp(x - max)
    exp_shifted = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_shifted)
    # logsumexp
    logsum = max_val + tl.log(sum_exp)
    # Output offset
    out_offset = b * (1 * h * w) + 0 * (h * w) + hi * w + wi
    tl.store(out_ptr + out_offset, logsum)


def triton_logsumexp(x: torch.Tensor, dim=1, keepdim=True):
    assert dim == 1 and keepdim, "Only dim=1 with keepdim=True supported"
    assert x.dim() == 4, "Input must be 4D"
    batch, channels, h, w = x.shape
    x = x.contiguous()
    out = torch.empty(batch, 1, h, w, dtype=x.dtype, device=x.device)
    grid = (batch * h * w,)
    logsumexp_kernel[grid](x, out, batch, channels, h, w)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, applies Group Normalization, fused Tanh+HardSwish+Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Fused Tanh + HardSwish + Residual Addition
        x_res = triton_fused(x_norm, x_conv)
        # LogSumExp
        x_logsumexp = triton_logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp