import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_scale_bias_relu_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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

        offs_k += BLOCK_SIZE_K

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Apply scale
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = acc * scale[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store output
    out_idx = offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def triton_gemm_scale_bias_relu(x, w, b, scale):
    assert x.is_cuda and w.is_cuda and b.is_cuda and scale.is_cuda
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    gemm_scale_bias_relu_kernel[grid](
        x, w, b, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out


@triton.jit
def fused_bn_kernel(
    x_ptr, mean_ptr, inv_std_ptr, weight_ptr, bias_ptr, out_ptr,
    num_rows, num_cols,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = cols < num_cols

    x = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask)
    mean = tl.load(mean_ptr + cols, mask=mask)
    inv_std = tl.load(inv_std_ptr + cols, mask=mask)
    weight = tl.load(weight_ptr + cols, mask=mask)
    bias = tl.load(bias_ptr + cols, mask=mask)

    out = (x - mean) * inv_std * weight + bias

    tl.store(out_ptr + row * stride_outm + cols * stride_outn, out, mask=mask)


def triton_fused_bn(x, mean, inv_std, weight, bias):
    assert x.is_cuda and mean.is_cuda and inv_std.is_cuda and weight.is_cuda and bias.is_cuda
    num_rows, num_cols = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = 128
    grid = (num_rows, triton.cdiv(num_cols, BLOCK_SIZE))

    fused_bn_kernel[grid](
        x, mean, inv_std, weight, bias, out,
        num_rows, num_cols,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = triton_gemm_scale_bias_relu(x, self.gemm.weight.T, self.gemm.bias, self.scale)
        mean = self.bn.running_mean
        var = self.bn.running_var
        inv_std = 1.0 / torch.sqrt(var + self.bn.eps)
        x = triton_fused_bn(x, mean, inv_std, self.bn.weight, self.bn.bias)
        return x