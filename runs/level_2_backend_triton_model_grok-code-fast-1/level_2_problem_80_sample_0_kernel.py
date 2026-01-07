import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    max_val = -float('inf')
    for i in range(0, out_features, BLOCK_SIZE):
        offsets = pid * out_features + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (pid + 1) * out_features
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x))
    tl.store(out_ptr + pid, max_val)


def triton_max(x: torch.Tensor, dim: int, keepdim: bool = False):
    assert x.is_cuda, "Tensor must be on CUDA."
    assert dim == 1, "Only dim=1 supported."
    x = x.contiguous()
    batch_size, out_features = x.shape
    out = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 128
    grid = (batch_size,)
    max_kernel[grid](x, out, batch_size, out_features, BLOCK_SIZE=BLOCK_SIZE)
    if keepdim:
        out = out.unsqueeze(1)
    return out


@triton.jit
def subtract_gelu_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr)
    out = x - mean
    # GELU approximation
    out = 0.5 * out * (1 + tl.tanh(0.7978845608028654 * (out + 0.044715 * out * out * out)))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_subtract_gelu(x: torch.Tensor, mean: torch.Tensor):
    assert x.is_cuda and mean.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    mean = mean.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    subtract_gelu_kernel[grid](x, mean, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = self.gemm(x)
        x = triton_max(x, dim=self.max_dim, keepdim=True)
        mean_val = x.mean(dim=0, keepdim=True)
        x = triton_subtract_gelu(x, mean_val)
        return x