import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_ptrs = w_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        mask_x = (rm[:, None] < M) & (k + rk[None, :] < K)
        mask_w = (k + rk[:, None] < K) & (rn[None, :] < N)
        x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_val = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(x_val, w_val)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    if b_ptr is not None:
        b_val = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
        acc = acc + b_val[None, :]

    # Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # First Mish
    acc_mish1 = acc
    sp1 = tl.log(1 + tl.exp(acc_mish1))
    tanh1 = tl.tanh(sp1)
    mish1 = acc_mish1 * tanh1

    # Second Mish
    sp2 = tl.log(1 + tl.exp(mish1))
    tanh2 = tl.tanh(sp2)
    mish2 = mish1 * tanh2

    out_ptrs = out_ptr + (rm[:, None] * stride_outm + rn[None, :] * stride_outn)
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, mish2, mask=mask_out)


def triton_linear_mish_mish(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    assert x.is_cuda and w.is_cuda and (b is None or b.is_cuda)
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    linear_mish_mish_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        w = self.linear.weight.T.contiguous()
        b = self.linear.bias
        return triton_linear_mish_mish(x, w, b)