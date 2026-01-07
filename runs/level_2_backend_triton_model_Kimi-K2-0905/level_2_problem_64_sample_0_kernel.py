import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_logsumexp_leaky_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk, mask=a_mask, other=0.0)
        b = tl.load(w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # LogSumExp along N dimension
    max_val = tl.max(acc, axis=1)
    max_val = tl.where(offs_m < M, max_val, float('-inf'))
    exp_sum = tl.sum(tl.exp(acc - max_val[:, None]), axis=1)
    logsumexp = max_val + tl.log(exp_sum)
    logsumexp = logsumexp[:, None]

    # LeakyReLU
    leaky1 = tl.where(logsumexp > 0, logsumexp, 0.01 * logsumexp)
    leaky2 = tl.where(leaky1 > 0, leaky1, 0.01 * leaky1)

    # GELU
    def gelu(x):
        cdf = 0.5 * (1.0 + tl.erf(x * 0.7071067811865475))
        return x * cdf
    gelu1 = gelu(leaky2)
    gelu2 = gelu(gelu1)

    out_offs = offs_m[:, None] * stride_outm + tl.zeros((BLOCK_M, 1), dtype=tl.int32) * stride_outn
    tl.store(out_ptr + out_offs, gelu2, mask=offs_m[:, None] < M)


def triton_fused_gemm_logsumexp_leaky_gelu(x, w, b):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    M, K = x.shape
    N = w.shape[0]
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    out = torch.empty((M, 1), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fused_gemm_logsumexp_leaky_gelu_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HAS_BIAS=True,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return triton_fused_gemm_logsumexp_leaky_gelu(x, self.linear.weight, self.linear.bias)