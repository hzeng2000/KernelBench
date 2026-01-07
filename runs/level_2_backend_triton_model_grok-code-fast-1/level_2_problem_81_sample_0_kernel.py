import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(
    a_ptr,  # Input x: (M, K)
    b_ptr,  # Weight: (N, K)
    bias_ptr,  # Bias: (N,)
    c_ptr,  # Output: (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M and offs_k[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K and offs_bn[None, :] < N, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # Add bias
    bias_vals = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    accumulator += bias_vals[None, :]
    # Swish: x * sigmoid(x)
    accumulator = accumulator * tl.sigmoid(accumulator)
    # Divide by 2
    accumulator = accumulator * 0.5
    # Clamp to [-1, 1]
    accumulator = tl.clamp(accumulator, -1.0, 1.0)
    # Tanh
    accumulator = tl.tanh(accumulator)
    # Clamp to [-1, 1]
    accumulator = tl.clamp(accumulator, -1.0, 1.0)
    # Store
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=offs_am[:, None] < M and offs_bn[None, :] < N)


def triton_fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device='cuda')
    grid = lambda meta: (tl.cdiv(M, meta['BLOCK_SIZE_M']), tl.cdiv(N, meta['BLOCK_SIZE_N']))
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    fused_linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return triton_fused_linear(x, self.gemm.weight, self.gemm.bias)