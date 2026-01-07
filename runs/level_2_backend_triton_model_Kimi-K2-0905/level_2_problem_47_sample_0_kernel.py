import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    softplus = tl.log(1 + tl.exp(x))
    tanh_sp = tl.tanh(softplus)
    out = x * tanh_sp
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.tanh(x)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def mish_tanh_fused_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # mish(x) = x * tanh(softplus(x))
    softplus = tl.log(1 + tl.exp(x))
    tanh_sp = tl.tanh(softplus)
    mish_out = x * tanh_sp
    # tanh(mish(x))
    out = tl.tanh(mish_out)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish_tanh_fused(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mish_tanh_fused_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    Uses fused Triton kernel for Mish+Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = triton_mish_tanh_fused(x)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]