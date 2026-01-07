import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_bias_kernel(
    x_ptr,  # Pointer to input x (batch_size, out_features)
    bias_ptr,  # Pointer to bias (1,)
    out_ptr,  # Pointer to output (batch_size, 1)
    n_elements,  # out_features
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    min_val = tl.reduce(x, lambda a, b: tl.minimum(a, b))
    bias = tl.load(bias_ptr)
    tl.store(out_ptr + pid, min_val + bias)


def triton_min_bias(x: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    batch_size, out_features = x.shape
    out = torch.empty(batch_size, 1, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = out_features
    grid = (batch_size,)
    min_bias_kernel[grid](x, bias, out, out_features, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = triton_min_bias(x, self.bias)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]