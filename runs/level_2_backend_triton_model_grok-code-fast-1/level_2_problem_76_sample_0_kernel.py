import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_relu_kernel(
    x_ptr,  # input: (batch_size, in_features)
    w_ptr,  # weight: (out_features, in_features)
    b_ptr,  # bias: (out_features,)
    out_ptr,  # output: (batch_size, out_features)
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,  # for batch
    BLOCK_SIZE_N: tl.constexpr,  # for out_features
    BLOCK_SIZE_K: tl.constexpr,  # for in_features
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_features, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
    w_ptrs = w_ptr + (offs_k[:, None] * out_features + offs_n[None, :])
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < batch_size and offs_k[None, :] < in_features, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < in_features and offs_n[None, :] < out_features, other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K * out_features
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < out_features, other=0.0)
    accumulator += b[None, :]
    # Apply ReLU
    accumulator = tl.maximum(accumulator, 0.0)
    # Store
    out_ptrs = out_ptr + offs_m[:, None] * out_features + offs_n[None, :]
    tl.store(out_ptrs, accumulator, mask=offs_m[:, None] < batch_size and offs_n[None, :] < out_features)


def fused_linear_relu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    batch_size, in_features = x.shape
    out_features = w.shape[0]
    out = torch.empty((batch_size, out_features), dtype=torch.float32, device='cuda')
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE_M']) * triton.cdiv(out_features, meta['BLOCK_SIZE_N']), )
    fused_linear_relu_kernel[grid](
        x, w, b, out,
        batch_size, in_features, out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        # Fused GEMM + bias + ReLU using Triton kernel
        return fused_linear_relu(x, self.gemm.weight, self.bias)