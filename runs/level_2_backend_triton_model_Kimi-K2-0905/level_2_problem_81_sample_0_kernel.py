import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_swish_div_clamp_tanh_clamp_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to blocks
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_x = (offs_m < M)[:, None] & (offs_k < K)[None, :]
        mask_w = (offs_k < K)[:, None] & (offs_n < N)[None, :]

        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc += tl.dot(x, w, trans_b=True)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    if HAS_BIAS:
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        b = tl.load(b_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        acc = acc + b[None, :]

    # Swish: x * sigmoid(x)
    sig = tl.sigmoid(acc)
    acc = acc * sig

    # Divide by 2
    acc = acc * 0.5

    # Clamp: min=-1.0, max=1.0
    acc = tl.clamp(acc, -1.0, 1.0)

    # Tanh
    acc = tl.tanh(acc)

    # Clamp again
    acc = tl.clamp(acc, -1.0, 1.0)

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (offs_out_m[:, None] * stride_outm + offs_out_n[None, :] * stride_outn)
    mask_out = (offs_out_m < M)[:, None] & (offs_out_n < N)[None, :]

    tl.store(out_ptrs, acc, mask=mask_out)


def triton_fused_gemm_swish_div_clamp_tanh_clamp(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: ((triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N)))

    fused_gemm_swish_div_clamp_tanh_clamp_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=bias is not None,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        weight = self.gemm.weight
        bias = self.gemm.bias
        return triton_fused_gemm_swish_div_clamp_tanh_clamp(x, weight, bias)