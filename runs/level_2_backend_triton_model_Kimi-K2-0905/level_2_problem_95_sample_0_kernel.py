import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, add_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bm,
    stride_addn,
    stride_om, stride_on,
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
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(x_ptr + a_idx, mask=a_mask, other=0.0)
        b = tl.load(w_ptr + b_idx, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c = acc
    # Add bias
    bias_offs = offs_n
    bias_mask = bias_offs < N
    bias = tl.load(b_ptr + bias_offs, mask=bias_mask, other=0.0)
    c += bias[None, :]

    # Add add_value
    add_offs = offs_n
    add_mask = add_offs < N
    add_val = tl.load(add_ptr + add_offs, mask=add_mask, other=0.0)
    c += add_val[None, :]

    # Swish: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-c))
    c = c * sig

    # Tanh
    c = tl.tanh(c)

    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gelu_c = c
    tanh_arg = 0.7978845608 * (gelu_c + 0.044715 * gelu_c * gelu_c * gelu_c)
    gelu_out = 0.5 * gelu_c * (1.0 + tl.tanh(tanh_arg))
    c = gelu_out

    # Hardtanh: clamp to [-1, 1]
    c = tl.where(c < -1.0, -1.0, c)
    c = tl.where(c > 1.0, 1.0, c)

    # Store output
    out_idx = (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_idx, c, mask=out_mask)


def triton_fused_forward(x, w, b, add_val):
    assert x.is_cuda and w.is_cuda and b.is_cuda and add_val.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    add_val = add_val.contiguous()

    M, K = x.shape
    N = w.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    fused_kernel[grid](
        x, w, b, add_val,
        out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0) if b.dim() > 0 else 0,
        add_val.stride(0) if add_val.dim() > 0 else 0,
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        return triton_fused_forward(x, self.matmul.weight, self.matmul.bias, self.add_value)