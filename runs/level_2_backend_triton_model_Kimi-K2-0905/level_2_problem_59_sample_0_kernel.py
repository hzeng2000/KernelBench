import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_scale_kernel(
    x_ptr,  # pointer to input/output
    n_elements,  # number of elements
    scaling_factor,  # scaling factor
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x * scaling_factor
    tl.store(x_ptr + offsets, out, mask=mask)


def fused_swish_scale(x: torch.Tensor, scaling_factor: float):
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    swish_scale_kernel[grid](x, n_elements, scaling_factor, BLOCK_SIZE=BLOCK_SIZE)
    return x


class ModelNew(nn.Module):
    """
    Optimized model that fuses Swish activation and scaling into a single Triton kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        return fused_swish_scale(x, self.scaling_factor)