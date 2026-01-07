import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_bias_sigmoid_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    bias = tl.load(bias_ptr + offs_bn * stride_bias, mask=offs_bn < N, other=0.0)
    c = accumulator + bias[None, :]
    c = tl.sigmoid(c)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, c, mask=c_mask)


@triton.jit
def fused_matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    bias = tl.load(bias_ptr + offs_bn * stride_bias, mask=offs_bn < N, other=0.0)
    c = accumulator + bias[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, c, mask=c_mask)


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr, M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x_ptrs = x_ptr + pid * stride_xm + offs * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    max_val = tl.max(x)
    shifted = x - max_val
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted)
    logsum = tl.log(sum_exp) + max_val
    tl.store(out_ptr + pid, logsum)


def triton_linear_sigmoid(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    fused_matmul_bias_sigmoid_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
        bias.stride(0),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


def triton_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    fused_matmul_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
        bias.stride(0),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


def triton_logsumexp(x):
    M, N = x.shape
    out = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_SIZE = N
    grid = (M,)
    logsumexp_kernel[grid](x, out, M, N, x.stride(0), x.stride(1), BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = triton_linear_sigmoid(x, self.linear1.weight, self.linear1.bias)
        x = triton_linear(x, self.linear2.weight, self.linear2.bias)
        x = triton_logsumexp(x)
        return x