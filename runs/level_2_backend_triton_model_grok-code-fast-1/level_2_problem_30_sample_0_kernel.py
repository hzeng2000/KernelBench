import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # add bias
    bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N)
    accumulator += bias_vals[None, :]
    tl.store(c_ptrs, accumulator, mask=mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    linear_kernel[grid](
        x, weight, out, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return out


@triton.jit
def groupnorm_hardtanh_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, C, G,
    eps: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c_start = g * (C // G)
    offsets = c_start + tl.arange(0, BLOCK_SIZE)
    x_row_ptr = x_ptr + n * C
    x_vals = tl.load(x_row_ptr + offsets)
    sum_x = tl.sum(x_vals)
    sum_x2 = tl.sum(x_vals * x_vals)
    num = BLOCK_SIZE
    mean = sum_x / num
    var = sum_x2 / num - mean * mean
    w_vals = tl.load(weight_ptr + offsets)
    b_vals = tl.load(bias_ptr + offsets)
    normalized = (x_vals - mean) / tl.sqrt(var + eps)
    out_vals = w_vals * normalized + b_vals
    out_vals = tl.maximum(out_vals, min_val)
    out_vals = tl.minimum(out_vals, max_val)
    out_row_ptr = out_ptr + n * C
    tl.store(out_row_ptr + offsets, out_vals)


def triton_groupnorm_hardtanh(x: torch.Tensor, gn_weight: torch.Tensor, gn_bias: torch.Tensor, num_groups: int, min_val: float, max_val: float):
    N, C = x.shape
    G = num_groups
    assert C % G == 0
    out = torch.empty_like(x)
    BLOCK_SIZE = C // G
    grid = (N, G)
    eps = 1e-5
    groupnorm_hardtanh_kernel[grid](
        x, gn_weight, gn_bias, out,
        N, C, G,
        eps, min_val, max_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = triton_linear(x, self.gemm.weight, self.gemm.bias)
        x = triton_groupnorm_hardtanh(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups, self.hardtanh.min_val, self.hardtanh.max_val)
        return x