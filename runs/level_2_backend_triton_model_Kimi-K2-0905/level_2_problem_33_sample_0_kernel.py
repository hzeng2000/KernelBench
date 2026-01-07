import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_scale_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bn,
    stride_scale,
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
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc += tl.dot(x, w)
        
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n * stride_bn, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]
    
    scale = tl.load(scale_ptr + offs_n * stride_scale, mask=offs_n < N, other=1.0)
    acc = acc * scale[None, :]
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def fused_bn_kernel(
    x_ptr, out_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, eps,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    
    mean = tl.load(mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=offs_n < N, other=1.0)
    weight = tl.load(weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    
    inv_std = tl.rsqrt(var + eps)
    
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offs_m_cur = m * BLOCK_SIZE_M + offs_m
        mask_m = offs_m_cur < M
        
        x_ptrs = x_ptr + offs_m_cur[:, None] * stride_xm + offs_n[None, :] * stride_xn
        mask = mask_m[:, None] & (offs_n[None, :] < N)
        
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        x_norm = (x - mean[None, :]) * inv_std[None, :]
        out = x_norm * weight[None, :] + bias[None, :]
        
        out_ptrs = out_ptr + offs_m_cur[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, out, mask=mask)


def triton_gemm_scale(x, w, b, scale):
    assert x.is_cuda and w.is_cuda and scale.is_cuda
    if b is not None:
        assert b.is_cuda
    
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w
    
    x = x.contiguous()
    w = w.contiguous()
    scale = scale.contiguous()
    if b is not None:
        b = b.contiguous()
    
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    
    gemm_scale_kernel[grid](
        x, w, b, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0) if b is not None else 0,
        scale.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out


def triton_fused_bn(x, mean, var, weight, bias, eps):
    assert x.is_cuda and mean.is_cuda and var.is_cuda and weight.is_cuda and bias.is_cuda
    
    M, N = x.shape
    x = x.contiguous()
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    fused_bn_kernel[grid](
        x, out, mean, var, weight, bias, eps,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = triton_gemm_scale(x, self.gemm.weight.t(), self.gemm.bias, self.scale)
        x = triton_fused_bn(x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, self.bn.eps)
        return x