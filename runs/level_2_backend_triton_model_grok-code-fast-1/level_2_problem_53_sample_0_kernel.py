import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    scaling_factor,  # Scaling factor
    min_val,  # Hardtanh min
    max_val,  # Hardtanh max
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Scaling
    x = x * scaling_factor
    # Hardtanh (clamp)
    x = tl.clamp(x, min_val, max_val)
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = tl.sqrt(2.0)
    erf_arg = x / sqrt_2
    erf_val = tl.libdevice.erf(erf_arg)
    x = 0.5 * x * (1.0 + erf_val)
    # Store output
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_post(x: torch.Tensor, scaling_factor: float, min_val: float, max_val: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_post_kernel[grid](x, out, scaling_factor, min_val, max_val, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.gemm(x)
        x = fused_post(x, self.scaling_factor, self.hardtanh_min, self.hardtanh_max)
        return x