import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_subtract_gelu_kernel(
    x_ptr, w_ptr, b_ptr, s_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bn,
    stride_sn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        x_idx = (offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :] * stride_xk)
        w_idx = ((k_start + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        mask_x = (offs_m < M)[:, None] & (k_start + offs_k < K)[None, :]
        mask_w = (k_start + offs_k < K)[:, None] & (offs_n < N)[None, :]
        x_block = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)
        w_block = tl.load(w_ptr + w_idx, mask=mask_w, other=0.0)
        acc += tl.dot(x_block, w_block)

    if b_ptr is not None:
        b_idx = offs_n * stride_bn
        mask_b = offs_n < N
        b = tl.load(b_ptr + b_idx, mask=mask_b, other=0.0)
        acc = acc + b[None, :]

    s_idx = offs_n * stride_sn
    mask_s = offs_n < N
    s = tl.load(s_ptr + s_idx, mask=mask_s, other=0.0)
    acc = acc - s[None, :]

    acc_gelu = 0.5 * acc * (1.0 + tl.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

    out_idx = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(out_ptr + out_idx, acc_gelu, mask=mask_out)


@triton.jit
def global_avg_pool_logsumexp_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_SIZE)
    mask_n = offs_n < N

    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_SIZE):
        idx = (pid_m * stride_xm + (n_start + offs_n) * stride_xn)
        mask = (pid_m < M) & mask_n
        x_val = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
        sum_val = tl.where(mask, sum_val + x_val, sum_val)

    total = tl.sum(sum_val, axis=0)
    avg = total / N

    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_SIZE):
        idx = (pid_m * stride_xm + (n_start + offs_n) * stride_xn)
        mask = (pid_m < M) & mask_n
        x_val = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, x_val)

    max_red = tl.max(max_val, axis=0)
    exp_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_SIZE):
        idx = (pid_m * stride_xm + (n_start + offs_n) * stride_xn)
        mask = (pid_m < M) & mask_n
        x_val = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
        exp_val = tl.exp(x_val - max_red)
        exp_sum = tl.where(mask, exp_sum + exp_val, exp_sum)

    lse = max_red + tl.log(tl.sum(exp_sum, axis=0))

    out_idx = pid_m * stride_om
    tl.store(out_ptr + out_idx, lse)


@triton.jit
def residual_add_kernel(
    x_ptr, res_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    res = tl.load(res_ptr + offsets, mask=mask)
    out = x + res
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_gemm_subtract_gelu(x, w, b, s):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    fused_gemm_subtract_gelu_kernel[grid](
        x, w, b, s, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        b.stride(0) if b is not None else 0,
        s.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


def triton_global_avg_pool_logsumexp(x):
    M, N = x.shape
    out = torch.empty((M, 1), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128
    grid = (M,)

    global_avg_pool_logsumexp_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_residual_add(x, res):
    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 128
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    residual_add_kernel[grid](
        x.view(-1), res.view(-1), out.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        x = triton_fused_gemm_subtract_gelu(x, self.gemm.weight, self.gemm.bias, self.subtract)
        x = triton_global_avg_pool_logsumexp(x)
        x = triton_residual_add(x, original_x)
        return x