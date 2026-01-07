import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr,  # Pointer to x (batch, out_features)
    s_ptr,  # Pointer to subtract (out_features,)
    mean_x_ptr,  # Pointer to mean_x (batch, 1)
    out_ptr,  # Pointer to output (batch, 1)
    batch_size,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    sum_val = 0.0
    for start in tl.range(0, out_features, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < out_features
        x_val = tl.load(x_ptr + b * out_features + offsets, mask=mask, other=0.0)
        s_val = tl.load(s_ptr + offsets, mask=mask, other=0.0)
        diff = x_val - s_val
        sum_val += tl.sum(diff)
    mean_val = sum_val / out_features
    gelu_val = tl.math.gelu(mean_val)
    mean_x_val = tl.load(mean_x_ptr + b)
    out_val = gelu_val + mean_x_val
    tl.store(out_ptr + b, out_val)


def triton_fused(x: torch.Tensor, s: torch.Tensor, mean_x: torch.Tensor):
    assert x.is_cuda and s.is_cuda and mean_x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    s = s.contiguous()
    mean_x = mean_x.contiguous()
    batch_size, out_features = x.shape
    out = torch.empty((batch_size, 1), dtype=torch.float32, device=x.device)
    BLOCK_SIZE = 128
    grid = (batch_size,)
    fused_kernel[grid](x, s, mean_x, out, batch_size, out_features, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        mean_x = torch.mean(original_x, dim=1, keepdim=True)
        # Gemm
        x = self.gemm(x)
        # Fused Subtract, GlobalAvgPool, LogSumExp (identity), GELU, ResidualAdd
        x = triton_fused(x, self.subtract, mean_x)
        return x