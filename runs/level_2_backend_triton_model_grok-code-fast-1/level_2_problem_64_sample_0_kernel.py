import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def logsumexp_activations_kernel(
    x_ptr,  # input (batch, out_features)
    out_ptr,  # output (batch, 1)
    batch_size,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    row_start = pid * out_features
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * out_features
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.reduce(x, tl.maximum, axis=0)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.reduce(exp_x, tl.add, axis=0)
    lse = tl.log(sum_exp) + max_val
    # leakyrelu
    lse = tl.where(lse > 0, lse, 0.01 * lse)
    # leakyrelu again
    lse = tl.where(lse > 0, lse, 0.01 * lse)
    # gelu
    sqrt2 = tl.sqrt(2.0)
    erf_arg = lse / sqrt2
    erf_val = tl.erf(erf_arg)
    gelu_val = 0.5 * lse * (1 + erf_val)
    lse = gelu_val
    # gelu again
    erf_arg = lse / sqrt2
    erf_val = tl.erf(erf_arg)
    gelu_val = 0.5 * lse * (1 + erf_val)
    lse = gelu_val
    tl.store(out_ptr + pid, lse)


def triton_logsumexp_activations(x: torch.Tensor):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    batch_size, out_features = x.shape
    out = torch.empty(batch_size, 1, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = out_features
    grid = (batch_size,)
    logsumexp_activations_kernel[grid](x, out, batch_size, out_features, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by fused LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations using Triton kernels.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # Fused LogSumExp + activations
        x = triton_logsumexp_activations(x)
        return x