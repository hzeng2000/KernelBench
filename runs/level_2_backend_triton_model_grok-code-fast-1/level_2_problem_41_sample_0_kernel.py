import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_normalize_gelu_relu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    eps,
    n_elements,
    features,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    feature_idx = offsets % features
    x = tl.load(x_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr + feature_idx, mask=mask)
    var = tl.load(var_ptr + feature_idx, mask=mask)
    gamma = tl.load(gamma_ptr + feature_idx, mask=mask)
    beta = tl.load(beta_ptr + feature_idx, mask=mask)
    normalized = (x - mean) / tl.sqrt(var + eps) * gamma + beta
    gelu_out = 0.5 * normalized * (1 + tl.erf(normalized / tl.sqrt(2.0)))
    out = tl.maximum(gelu_out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_normalize_gelu_relu(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float):
    assert x.is_cuda and mean.is_cuda and var.is_cuda and gamma.is_cuda and beta.is_cuda
    x = x.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    features = x.shape[-1]
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_normalize_gelu_relu_kernel[grid](
        x, mean, var, gamma, beta, out, eps, n_elements, features, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.batch_norm.running_mean = (
                self.batch_norm.momentum * self.batch_norm.running_mean + (1 - self.batch_norm.momentum) * mean
            )
            self.batch_norm.running_var = (
                self.batch_norm.momentum * self.batch_norm.running_var + (1 - self.batch_norm.momentum) * var
            )
        else:
            mean = self.batch_norm.running_mean
            var = self.batch_norm.running_var
        x = fused_normalize_gelu_relu(
            x, mean, var, self.batch_norm.weight, self.batch_norm.bias, self.batch_norm.eps
        )
        return x