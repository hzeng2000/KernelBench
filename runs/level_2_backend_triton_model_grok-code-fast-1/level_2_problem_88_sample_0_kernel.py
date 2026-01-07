import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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
    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N)
    accumulator += bias[None, :]
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=offs_cm[:, None] < M and offs_cn[None, :] < N)


def triton_matmul(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    assert a.is_cuda and b.is_cuda and bias.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    matmul_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return c


@triton.jit
def fused_groupnorm_swish_kernel(
    x_ptr, out_ptr,
    weight_ptr, bias_ptr, multiply_weight_ptr,
    batch_size, out_features, num_groups, group_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // num_groups
    g = pid % num_groups
    offsets = b * out_features + g * group_size + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < group_size
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    sum_x = tl.sum(x_vals)
    sum_x2 = tl.sum(x_vals * x_vals)
    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    weight_offsets = g * group_size + tl.arange(0, BLOCK_SIZE)
    weight_vals = tl.load(weight_ptr + weight_offsets, mask=mask)
    bias_vals = tl.load(bias_ptr + weight_offsets, mask=mask)
    multiply_vals = tl.load(multiply_weight_ptr + weight_offsets, mask=mask)
    normalized = (x_vals - mean) / tl.sqrt(var + eps)
    out_vals = weight_vals * normalized + bias_vals
    sig = 1 / (1 + tl.exp(-out_vals))
    out_vals = out_vals * sig
    out_vals = out_vals * multiply_vals
    sig = 1 / (1 + tl.exp(-out_vals))
    out_vals = out_vals * sig
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def triton_fused_groupnorm_swish(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, multiply_weight: torch.Tensor, num_groups: int, eps: float = 1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and multiply_weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    multiply_weight = multiply_weight.contiguous()
    out = torch.empty_like(x)
    batch_size, out_features = x.shape
    group_size = out_features // num_groups
    grid = (batch_size * num_groups,)
    fused_groupnorm_swish_kernel[grid](
        x, out, weight, bias, multiply_weight,
        batch_size, out_features, num_groups, group_size,
        eps, BLOCK_SIZE=group_size
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape)) 

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = triton_matmul(x, self.gemm.weight.t(), self.gemm.bias)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = triton_fused_groupnorm_swish(x, self.group_norm.weight, self.group_norm.bias, self.multiply_weight, self.group_norm.num_groups)
        return x