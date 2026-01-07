import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def subtract_hardswish_kernel(
    x_ptr,  # Pointer to input
    subtract_value,  # Scalar to subtract
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = x - subtract_value
    # HardSwish: x * clamp(x + 3, 0, 6) / 6
    clamped = tl.clamp(x + 3, 0, 6)
    out = x * clamped / 6
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_subtract_hardswish(x: torch.Tensor, subtract_value: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    subtract_hardswish_kernel[grid](x, subtract_value, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def mish_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Stable softplus: if x > 20, softplus = x, else log(1 + exp(x))
    exp_x = tl.exp(tl.where(x > 20, 0, x))
    softplus = tl.where(x > 20, x, tl.log(1 + exp_x))
    tanh_softplus = tl.tanh(softplus)
    out = x * tanh_softplus
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_mish(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    Optimized with Triton kernels for subtraction + HardSwish fusion and Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = triton_subtract_hardswish(x, self.subtract_value)
        x = self.pool(x)
        x = triton_mish(x)
        return x