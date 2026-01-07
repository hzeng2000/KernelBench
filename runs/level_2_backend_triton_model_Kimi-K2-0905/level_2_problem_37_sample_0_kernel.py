import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_swish_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bn,
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

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    b_ptrs = b_ptr + offs_n * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        mask_k = (k + offs_k)[None, :] < K
        mask_m = offs_m[:, None] < M
        x_mask = mask_m & mask_k
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        mask_n = offs_n[None, :] < N
        w_mask = mask_k[:, None] & mask_n
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    out_mask = mask_m & mask_n

    b = tl.load(b_ptrs, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    sig = tl.sigmoid(acc)
    out = sig * acc

    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, out, mask=out_mask)


@triton.jit
def group_norm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    M, N, num_groups, group_size,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_g * group_size + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.sum(x, axis=1) / group_size
    mean = mean[:, None]

    var = tl.sum((x - mean) * (x - mean), axis=1) / group_size
    var = var[:, None]

    x_norm = (x - mean) / tl.sqrt(var + 1e-5)

    weight_ptrs = weight_ptr + offs_n * stride_xn
    bias_ptrs = bias_ptr + offs_n * stride_xn

    weight = tl.load(weight_ptrs, mask=mask_n, other=1.0)
    bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)

    out = x_norm * weight[None, :] + bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask)


def triton_matmul_swish_bias(x, w, b):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: ((M + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"],
                         (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"])

    fused_matmul_swish_bias_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


def triton_group_norm(x, num_groups, weight, bias):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, N = x.shape
    assert N % num_groups == 0
    group_size = N // num_groups

    out = torch.empty_like(x)

    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = min(group_size, 128)

    grid = lambda META: (M, num_groups)

    group_norm_kernel[grid](
        x, out, weight, bias,
        M, N, num_groups, group_size,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        self.group_norm_weight = nn.Parameter(torch.ones(out_features))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = triton_matmul_swish_bias(x, self.weight.t(), self.bias)
        x = triton_group_norm(x, self.num_groups, self.group_norm_weight, self.group_norm_bias)
        return x