import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_bias_hardtanh_mish_gn_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, mean_ptr, rstd_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_ptrs = w_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_x = (rm[:, None] < M) & (rk[None, :] < K - k)
        mask_w = (rk[:, None] < K - k) & (rn[None, :] < N)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    acc = acc + b[None, :]

    # Hardtanh
    acc = tl.where(acc < -1, -1, acc)
    acc = tl.where(acc > 1, 1, acc)

    # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    exp_acc = tl.exp(acc)
    softplus = tl.log(1 + exp_acc)
    tanh_sp = tl.tanh(softplus)
    mish_out = acc * tanh_sp

    # Store intermediate output for group norm
    out_offsets = rm[:, None] * stride_outm + rn[None, :] * stride_outn
    tl.store(out_ptr + out_offsets, mish_out, mask=(rm[:, None] < M) & (rn[None, :] < N))

    # Compute mean and rstd for group norm
    gn_group_size = N // 256  # num_groups = 256
    group_id = rn // gn_group_size
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    mish_masked = tl.where(mask, mish_out, 0.0)

    # Compute mean per group
    for g in range(256):
        g_mask = (group_id == g) & mask
        g_count = tl.sum(g_mask.to(tl.float32))
        g_sum = tl.sum(tl.where(g_mask, mish_masked, 0.0))
        mean = g_sum / g_count
        var = tl.sum(tl.where(g_mask, (mish_masked - mean) ** 2, 0.0)) / g_count
        rstd = tl.rsqrt(var + eps)

        # Store mean and rstd
        mean_out_offset = (rm * 256 + g) * stride_outn
        rstd_out_offset = (rm * 256 + g) * stride_outn + 1
        tl.store(mean_ptr + mean_out_offset, mean, mask=rm < M)
        tl.store(rstd_ptr + rstd_out_offset, rstd, mask=rm < M)


@triton.jit
def groupnorm_apply_kernel(
    out_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr,
    M, N,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    gn_group_size = N // 256
    group_id = rn // gn_group_size

    out_offsets = rm[:, None] * stride_outm + rn[None, :] * stride_outn
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    x = tl.load(out_ptr + out_offsets, mask=mask, other=0.0)

    mean_offset = rm * 256 + group_id
    mean = tl.load(mean_ptr + mean_offset[:, None], mask=mask, other=0.0)
    rstd_offset = rm * 256 + group_id + 1
    rstd = tl.load(rstd_ptr + rstd_offset[:, None], mask=mask, other=0.0)

    gamma = tl.load(gamma_ptr + rn, mask=rn < N, other=1.0)
    beta = tl.load(beta_ptr + rn, mask=rn < N, other=0.0)

    x_norm = (x - mean) * rstd
    out = gamma[None, :] * x_norm + beta

    tl.store(out_ptr + out_offsets, out, mask=mask)


def fused_gemm_bias_hardtanh_mish_gn(x, w, b, num_groups, eps=1e-5):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K
    assert b.shape[0] == N

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    mean = torch.empty((M, num_groups), dtype=x.dtype, device=x.device)
    rstd = torch.empty((M, num_groups), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = lambda META: ((M + META["BLOCK_M"] - 1) // META["BLOCK_M"],
                         (N + META["BLOCK_N"] - 1) // META["BLOCK_N"])

    fused_gemm_bias_hardtanh_mish_gn_kernel[grid](
        x, w, b, out, mean, rstd,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    # Apply group norm scaling
    gamma = torch.ones(N, dtype=x.dtype, device=x.device)
    beta = torch.zeros(N, dtype=x.dtype, device=x.device)

    grid2 = lambda META: ((M + META["BLOCK_M"] - 1) // META["BLOCK_M"],
                          (N + META["BLOCK_N"] - 1) // META["BLOCK_N"])

    groupnorm_apply_kernel[grid2](
        out, mean, rstd, gamma, beta,
        M, N,
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups

    def forward(self, x):
        w = self.gemm.weight
        b = self.bias
        return fused_gemm_bias_hardtanh_mish_gn(x, w, b, self.num_groups)