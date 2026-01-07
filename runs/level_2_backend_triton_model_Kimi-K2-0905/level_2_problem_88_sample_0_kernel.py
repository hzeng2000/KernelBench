import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_groupnorm_swish_multiply_swish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, weight_ptr, mean_ptr, rstd_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
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

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(x_ptrs, mask=mask_x, other=0.0)
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(a, b)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # GroupNorm
    group_size = N // NUM_GROUPS
    group_id_n = offs_n // group_size
    acc_mean = tl.zeros((BLOCK_SIZE_M, NUM_GROUPS), dtype=tl.float32)
    acc_var = tl.zeros((BLOCK_SIZE_M, NUM_GROUPS), dtype=tl.float32)

    for g in range(NUM_GROUPS):
        g_start = g * group_size
        g_end = (g + 1) * group_size
        mask_g = (offs_n >= g_start) & (offs_n < g_end) & (offs_n < N)
        acc_g = tl.where(mask_g[None, :], acc, 0.0)
        count = tl.sum(tl.where(mask_g[None, :], 1.0, 0.0), axis=1)
        mean = tl.sum(acc_g, axis=1) / count
        var = tl.sum((acc_g - mean[:, None]) ** 2, axis=1) / count
        acc_mean[:, g] = mean
        acc_var[:, g] = var

    rstd = tl.rsqrt(acc_var + eps)
    norm_acc = tl.zeros_like(acc)
    for g in range(NUM_GROUPS):
        g_start = g * group_size
        g_end = (g + 1) * group_size
        mask_g = (offs_n >= g_start) & (offs_n < g_end) & (offs_n < N)
        norm_acc = tl.where(mask_g[None, :], (acc - acc_mean[:, g:g+1]) * rstd[:, g:g+1], norm_acc)

    # Swish
    sig = tl.sigmoid(norm_acc)
    swish = norm_acc * sig

    # Multiply
    weight = tl.load(weight_ptr + offs_n, mask=offs_n < N, other=0.0)
    mul = swish * weight[None, :]

    # Swish
    sig2 = tl.sigmoid(mul)
    out = mul * sig2

    out_ptrs = out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out, mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        M, K = x.shape
        N = self.gemm.out_features
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)

        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

        gemm_groupnorm_swish_multiply_swish_kernel[grid](
            x, self.gemm.weight, self.gemm.bias, out, self.multiply_weight, None, None,
            M, N, K,
            x.stride(0), x.stride(1),
            self.gemm.weight.stride(0), self.gemm.weight.stride(1),
            out.stride(0), out.stride(1),
            eps=self.eps,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            NUM_GROUPS=self.num_groups,
        )
        return out