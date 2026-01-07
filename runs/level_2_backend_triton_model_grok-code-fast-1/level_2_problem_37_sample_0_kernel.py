import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_swish_bias_kernel(
    x_ptr,  # (B, M)
    weight_ptr,  # (N, M)
    bias_linear_ptr,  # (N,)
    bias_ptr,  # (N,)
    out_ptr,  # (B, N)
    B, M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_b = pid_b * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, M, BLOCK_SIZE_K):
        offs_k_curr = k + offs_k
        mask_k = offs_k_curr < M
        mask_b = offs_b[:, None] < B
        mask_n = offs_n[:, None] < N

        x_vals = tl.load(x_ptr + offs_b[:, None] * M + offs_k_curr[None, :], mask=mask_b & mask_k[None, :], other=0.0)
        w_vals = tl.load(weight_ptr + offs_n[:, None] * M + offs_k_curr[None, :], mask=mask_n & mask_k[None, :], other=0.0)

        acc += tl.dot(x_vals, w_vals.T)

    # Add bias_linear
    bias_linear_vals = tl.load(bias_linear_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias_linear_vals[None, :]

    # Swish: z * sigmoid(z) = z / (1 + exp(-z))
    z = acc
    swish = z * (1.0 / (1.0 + tl.exp(-z)))

    # Add bias
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    swish += bias_vals[None, :]

    # Store
    tl.store(out_ptr + offs_b[:, None] * N + offs_n[None, :], swish, mask=(offs_b[:, None] < B) & (offs_n[None, :] < N))


def triton_fused_linear_swish_bias(weight: torch.Tensor, bias_linear: torch.Tensor, bias: torch.Tensor, x: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias_linear.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias_linear = bias_linear.contiguous()
    bias = bias.contiguous()

    B, M = x.shape
    N, M_ = weight.shape
    assert M == M_, "Shape mismatch"

    out = torch.empty(B, N, dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    grid = ((B + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    fused_linear_swish_bias_kernel[grid](
        x, weight, bias_linear, bias, out,
        B, M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_linear = nn.Parameter(torch.randn(out_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = triton_fused_linear_swish_bias(self.weight, self.bias_linear, self.bias, x)
        x = self.group_norm(x)
        return x