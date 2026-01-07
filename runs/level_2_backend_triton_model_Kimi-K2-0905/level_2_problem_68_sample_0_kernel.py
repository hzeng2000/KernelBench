import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_min_sub_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    constant,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
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
        a_idx = (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        b_idx = (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(x_ptr + a_idx, mask=mask_a, other=0.0)
        b = tl.load(w_ptr + b_idx, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    acc = tl.minimum(acc, constant)
    acc = acc - constant

    out_idx = (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_idx, acc, mask=mask_out)


def fused_linear_min_sub(x, w, b, constant):
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float32 and w.dtype == torch.float32
    if b is not None:
        assert b.dtype == torch.float32

    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w

    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    out = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),
                         triton.cdiv(N, META['BLOCK_SIZE_N']))

    fused_linear_min_sub_kernel[grid](
        x, w, b, out,
        M, K, N,
        constant,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = torch.tensor(constant, dtype=torch.float32)

    def forward(self, x):
        return fused_linear_min_sub(x, self.linear.weight.t(), self.linear.bias, self.constant)