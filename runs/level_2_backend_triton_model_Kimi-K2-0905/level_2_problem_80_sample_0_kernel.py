import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_max_sub_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, M, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        a = tl.load(x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk), mask=(rm[:, None] < B) & (rk[None, :] < N), other=0.0)
        w = tl.load(w_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn), mask=(rk[:, None] < N) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, w)

    if b_ptr is not None:
        b = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
        acc = acc + b[None, :]

    # Max along dim=1
    max_val = tl.max(acc, axis=1)
    acc = acc - max_val[:, None]

    # Subtract mean along dim=1
    mean_val = tl.sum(acc, axis=1) / N
    acc = acc - mean_val[:, None]

    # GELU
    acc = 0.5 * acc * (1.0 + tl.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

    tl.store(out_ptr + (rm[:, None] * stride_outm + rn[None, :] * stride_outn), acc, mask=(rm[:, None] < B) & (rn[None, :] < N))


def triton_gemm_max_sub_gelu(x, w, b):
    B, M = x.shape
    N, _ = w.shape
    out = torch.empty((B, N), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))

    gemm_max_sub_gelu_kernel[grid](
        x, w, b, out,
        B, M, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        w = self.gemm.weight
        b = self.gemm.bias
        return triton_gemm_max_sub_gelu(x, w, b)