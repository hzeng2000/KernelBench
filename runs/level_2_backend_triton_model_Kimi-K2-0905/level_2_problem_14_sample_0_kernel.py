import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_div_sum_scale_kernel(
    x_ptr, w_ptr, out_ptr,
    B, M, N, K,
    stride_xb, stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ob, stride_om,
    scaling_factor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + rk
        mask_x = (rm[:, None] < M) & (k_offs[None, :] < K)
        mask_w = (k_offs[:, None] < K) & (rn[None, :] < N)

        x_ptrs = x_ptr + pid_b * stride_xb + rm[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + k_offs[:, None] * stride_wk + rn[None, :] * stride_wn

        x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_val = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc += tl.sum(x_val[:, :, None] * w_val[None, :, :], axis=1)

    acc = acc / 2.0
    sum_val = tl.sum(acc, axis=0)
    out_val = sum_val * scaling_factor

    out_ptrs = out_ptr + pid_b * stride_ob + pid_m * stride_om
    tl.store(out_ptrs, out_val)


def triton_fused_gemm_div_sum_scale(x: torch.Tensor, w: torch.Tensor, scaling_factor: float):
    assert x.is_cuda and w.is_cuda
    B, M, K = x.shape[0], x.shape[1], x.shape[2]
    N = w.shape[0]
    x = x.contiguous()
    w = w.contiguous()

    out = torch.empty((B, 1), dtype=torch.float32, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), B)

    fused_gemm_div_sum_scale_kernel[grid](
        x, w, out,
        B, M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        scaling_factor,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.unsqueeze(1)
        w = self.weight.T.unsqueeze(0)
        return triton_fused_gemm_div_sum_scale(x, w, self.scaling_factor)