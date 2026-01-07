import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_sub_tanh_sub_kernel(
    x_ptr,  # Pointer to input (output of conv)
    out_ptr,  # Pointer to output
    sub1_val,  # Subtract1 value
    sub2_val,  # Subtract2 value
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.math.tanh(x - sub1_val) - sub2_val
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_sub_tanh_sub(x: torch.Tensor, sub1_val: float, sub2_val: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_sub_tanh_sub_kernel[grid](x, out, sub1_val, sub2_val, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, fused subtraction-tanh-subtraction, and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x)
        x = triton_fused_sub_tanh_sub(x, self.subtract1_value, self.subtract2_value)
        x = self.avgpool(x)
        return x