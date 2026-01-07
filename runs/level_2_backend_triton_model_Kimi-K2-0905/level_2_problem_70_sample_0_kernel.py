import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_sigmoid_scaling_residual_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    scaling_factor,
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
        k_start = k * BLOCK_SIZE_K
        x_idx = (offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :] * stride_xk)
        w_idx = ((k_start + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        mask_x = (offs_m[:, None] < M) & ((k_start + offs_k)[None, :] < K)
        mask_w = ((k_start + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        x_block = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)
        w_block = tl.load(w_ptr + w_idx, mask=mask_w, other=0.0)
        acc += tl.dot(x_block, w_block)

    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]

    sigmoid_out = tl.sigmoid(acc)
    scaled_out = sigmoid_out * scaling_factor
    residual_out = scaled_out + acc

    out_idx = (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_idx, residual_out, mask=mask_out)


def triton_gemm_sigmoid_scaling_residual(x, w, b, scaling_factor):
    assert x.is_cuda and w.is_cuda
    if b is not None:
        assert b.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    gemm_sigmoid_scaling_residual_kernel[grid](
        x, w, b, out,
        M, N, K,
        scaling_factor,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return triton_gemm_sigmoid_scaling_residual(x, self.gemm.weight.t(), self.gemm.bias, self.scaling_factor)