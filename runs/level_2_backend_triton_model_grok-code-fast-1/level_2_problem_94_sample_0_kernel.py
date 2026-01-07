import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_kernel(
    x_ptr, w_ptr, bias1_ptr, bias2_ptr, out_ptr,
    B, C, K,
    stride_xb, stride_xk,
    stride_wc, stride_wk,
    stride_ob, stride_oc,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(C, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xb + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wc

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < B and offs_k[None, :] < K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K and offs_n[None, :] < C, other=0.0)
        accumulator += tl.dot(x, w)
        offs_k += BLOCK_SIZE_K
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    bias1 = tl.load(bias1_ptr + offs_n, mask=offs_n < C, other=0.0)
    bias2 = tl.load(bias2_ptr + offs_n, mask=offs_n < C, other=0.0)
    accumulator += bias1[None, :] + bias2[None, :]

    accumulator = tl.clamp(accumulator, -1.0, 1.0)
    softplus = tl.log(1 + tl.exp(accumulator))
    accumulator = accumulator * tl.tanh(softplus)

    out_ptrs = out_ptr + offs_m[:, None] * stride_ob + offs_n[None, :] * stride_oc
    tl.store(out_ptrs, accumulator, mask=offs_m[:, None] < B and offs_n[None, :] < C)


def triton_fused_linear(x: torch.Tensor, weight: torch.Tensor, bias1: torch.Tensor, bias2: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias1.is_cuda and bias2.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias1 = bias1.contiguous()
    bias2 = bias2.contiguous()
    B, K = x.shape
    C, _ = weight.shape
    out = torch.empty(B, C, dtype=torch.float32, device=x.device)
    stride_xb, stride_xk = x.stride()
    stride_wc, stride_wk = weight.stride()
    stride_ob, stride_oc = out.stride()
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_SIZE_M"]) * triton.cdiv(C, meta["BLOCK_SIZE_N"]),)
    fused_linear_kernel[grid](
        x, weight, bias1, bias2, out,
        B, C, K,
        stride_xb, stride_xk,
        stride_wc, stride_wk,
        stride_ob, stride_oc,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M
    )
    return out


@triton.jit
def groupnorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C, num_groups, channels_per_group, eps
):
    pid = tl.program_id(0)
    b = pid // num_groups
    g = pid % num_groups
    offs_c = g * channels_per_group + tl.arange(0, channels_per_group)
    x_ptrs = x_ptr + b * C + offs_c
    x_vals = tl.load(x_ptrs, mask=offs_c < C, other=0.0)
    mean = tl.sum(x_vals) / channels_per_group
    var = tl.sum((x_vals - mean)**2) / channels_per_group
    x_norm = (x_vals - mean) / tl.sqrt(var + eps)
    w_vals = tl.load(weight_ptr + offs_c, mask=offs_c < C, other=1.0)
    b_vals = tl.load(bias_ptr + offs_c, mask=offs_c < C, other=0.0)
    out_vals = x_norm * w_vals + b_vals
    out_ptrs = out_ptr + b * C + offs_c
    tl.store(out_ptrs, out_vals, mask=offs_c < C)


def triton_groupnorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int, eps: float = 1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    B, C = x.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    out = torch.empty_like(x)
    grid = (B * num_groups,)
    groupnorm_kernel[grid](
        x, weight, bias, out,
        B, C, num_groups, channels_per_group, eps
    )
    return out


class ModelNew(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = triton_fused_linear(x, self.gemm.weight, self.gemm.bias, self.bias)
        x = triton_groupnorm(x, self.groupnorm.weight, self.groupnorm.bias, self.groupnorm.num_groups)
        return x