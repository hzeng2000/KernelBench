import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_scale_hardtanh_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_outm, stride_outn,
    scaling_factor,
    hardtanh_min, hardtanh_max,
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
        k_idx = k * BLOCK_SIZE_K + offs_k
        x_idx = offs_m[:, None] * stride_xm + k_idx[None, :]
        w_idx = offs_n[None, :] * stride_wn + k_idx[:, None]

        x_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        w_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
        w_val = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc += tl.dot(x_val, w_val)

    if b_ptr is not None:
        b_val = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += b_val[None, :]

    acc = acc * scaling_factor
    acc = tl.where(acc < hardtanh_min, hardtanh_min, acc)
    acc = tl.where(acc > hardtanh_max, hardtanh_max, acc)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pi = 3.141592653589793
    coeff = tl.sqrt(2.0 / pi)
    cube = acc * acc * acc
    inner = coeff * (acc + 0.044715 * cube)
    tanh_inner = tl.tanh(inner)
    gelu_out = 0.5 * acc * (1.0 + tanh_inner)

    out_idx = offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_idx, gelu_out, mask=out_mask)


def triton_gemm_scale_hardtanh_gelu(x, w, b, scaling_factor, hardtanh_min, hardtanh_max):
    assert x.is_cuda and w.is_cuda
    if b is not None:
        assert b.is_cuda
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    gemm_scale_hardtanh_gelu_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        scaling_factor,
        hardtanh_min, hardtanh_max,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, scaling, hardtanh, and GELU into a single Triton kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.gelu = nn.GELU()

    def forward(self, x):
        return triton_gemm_scale_hardtanh_gelu(
            x, self.gemm.weight.t(), self.gemm.bias,
            self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )