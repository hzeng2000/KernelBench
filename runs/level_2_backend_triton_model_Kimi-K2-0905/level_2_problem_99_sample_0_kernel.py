import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_gelu_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bm,
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
        k_start = k * BLOCK_SIZE_K
        x_idx = (offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :] * stride_xk)
        w_idx = ((k_start + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        mask_x = (offs_m < M)[:, None] & ((k_start + offs_k) < K)[None, :]
        mask_w = ((k_start + offs_k) < K)[:, None] & (offs_n < N)[None, :]
        x_block = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)
        w_block = tl.load(w_ptr + w_idx, mask=mask_w, other=0.0)
        acc += tl.dot(x_block, w_block)

    if b_ptr is not None:
        b_idx = offs_n * stride_bm
        mask_b = offs_n < N
        b = tl.load(b_ptr + b_idx, mask=mask_b, other=0.0)
        acc = acc + b[None, :]

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_gelu = acc
    coeff = tl.sqrt(2.0 / 3.141592653589793)
    x_cubed = x_gelu * x_gelu * x_gelu
    tanh_arg = coeff * (x_gelu + 0.044715 * x_cubed)
    # Approx tanh with a rational function: tanh(x) ≈ x / (1 + |x|) for small x, but we use full here
    # Simplified: use tanh intrinsic
    tanh_val = tl.tanh(tanh_arg)
    gelu_out = 0.5 * x_gelu * (1.0 + tanh_val)

    # Softmax along dim=1 (row-wise)
    # Subtract max for numerical stability
    max_val = tl.max(gelu_out, axis=1)
    max_val = max_val[:, None]
    exp_val = tl.exp(gelu_out - max_val)
    sum_val = tl.sum(exp_val, axis=1)
    sum_val = sum_val[:, None]
    softmax_out = exp_val / sum_val

    out_idx = (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    mask_out = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(out_ptr + out_idx, softmax_out, mask=mask_out)


def triton_linear_gelu_softmax(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == torch.float32
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    linear_gelu_softmax_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        bias.stride(0) if bias is not None else 0,
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return triton_linear_gelu_softmax(x, self.linear.weight, self.linear.bias)