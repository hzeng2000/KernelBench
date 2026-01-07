import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_sigmoid_gemm_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    M, K1, N1, K2, N2,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_b1n,
    stride_w2k, stride_w2n,
    stride_b2n,
    stride_outm,
    BLOCK_M: tl.constexpr, BLOCK_K1: tl.constexpr, BLOCK_N1: tl.constexpr,
    BLOCK_K2: tl.constexpr, BLOCK_N2: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # First GEMM + Sigmoid
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn1 = pid_n * BLOCK_N1 + tl.arange(0, BLOCK_N1)
    rm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rn1 = tl.max_contiguous(tl.multiple_of(rn1 % N1, BLOCK_N1), BLOCK_N1)
    rk1 = tl.arange(0, BLOCK_K1)

    acc1 = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)
    for k1 in range(0, K1, BLOCK_K1):
        k1_offs = k1 + rk1
        x_offs = rm[:, None] * stride_xm + k1_offs[None, :] * stride_xk
        w1_offs = k1_offs[:, None] * stride_w1k + rn1[None, :] * stride_w1n
        mask_x = (rm < M)[:, None] & (k1_offs < K1)[None, :]
        mask_w1 = (k1_offs < K1)[:, None] & (rn1 < N1)[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=mask_x, other=0.0)
        w1_tile = tl.load(w1_ptr + w1_offs, mask=mask_w1, other=0.0)
        acc1 += tl.dot(x_tile, w1_tile)

    # Add bias1
    b1_offs = rn1
    mask_b1 = rn1 < N1
    b1 = tl.load(b1_ptr + b1_offs, mask=mask_b1, other=0.0)
    acc1 += b1[None, :]

    # Sigmoid
    acc1 = tl.sigmoid(acc1)

    # Second GEMM
    rk2 = tl.arange(0, BLOCK_K2)
    rn2 = tl.arange(0, BLOCK_N2)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)
    for k2 in range(0, N1, BLOCK_K2):
        k2_offs = k2 + rk2
        a1_offs = rm[:, None] * stride_xm + k2_offs[None, :] * stride_xk
        w2_offs = k2_offs[:, None] * stride_w2k + rn2[None, :] * stride_w2n
        mask_a1 = (rm < M)[:, None] & (k2_offs < N1)[None, :]
        mask_w2 = (k2_offs < N1)[:, None] & (rn2 < N2)[None, :]
        a1_tile = acc1  # reuse acc1 as activation
        w2_tile = tl.load(w2_ptr + w2_offs, mask=mask_w2, other=0.0)
        acc2 += tl.dot(a1_tile, w2_tile)

    # Add bias2
    b2_offs = rn2
    mask_b2 = rn2 < N2
    b2 = tl.load(b2_ptr + b2_offs, mask=mask_b2, other=0.0)
    acc2 += b2[None, :]

    # Store output before LogSumExp
    out_offs = rm[:, None] * stride_outm + rn2[None, :]
    mask_out = (rm < M)[:, None] & (rn2 < N2)[None, :]
    tl.store(out_ptr + out_offs, acc2, mask=mask_out)


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr, M, N,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    offs = rm[:, None] * stride_xm + rn[None, :] * stride_xn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))

    # Online max and sum
    max_val = tl.max(x, axis=1)
    max_val = tl.where(tl.isnan(max_val), -float('inf'), max_val)
    x_shifted = x - max_val[:, None]
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=1)

    logsumexp = max_val + tl.log(sum_exp)
    out_offs = rm
    tl.store(out_ptr + out_offs, logsumexp, mask=rm < M)


def triton_gemm_sigmoid_gemm(x, w1, b1, w2, b2):
    M, K1 = x.shape
    N1, K2 = w1.shape
    N2, K2 = w2.shape
    assert K1 == K2

    out = torch.empty((M, N2), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_K1 = 32
    BLOCK_N1 = 32
    BLOCK_K2 = 32
    BLOCK_N2 = 32

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N1, META['BLOCK_N1']))

    gemm_sigmoid_gemm_kernel[grid](
        x, w1, b1, w2, b2, out,
        M, K1, N1, K2, N2,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        b1.stride(0),
        w2.stride(0), w2.stride(1),
        b2.stride(0),
        out.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_K1=BLOCK_K1, BLOCK_N1=BLOCK_N1,
        BLOCK_K2=BLOCK_K2, BLOCK_N2=BLOCK_N2,
    )
    return out


def triton_logsumexp(x, dim):
    assert dim == 1
    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

    logsumexp_kernel[grid](
        x, out, M, N,
        x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        w1 = self.linear1.weight.t()
        b1 = self.linear1.bias
        w2 = self.linear2.weight.t()
        b2 = self.linear2.bias
        x = triton_gemm_sigmoid_gemm(x, w1, b1, w2, b2)
        x = triton_logsumexp(x, dim=1)
        return x