import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_bn_gelu_relu_kernel(
    x_ptr, w_ptr, b_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr, out_gelu_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x_chunk = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_chunk = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_chunk, w_chunk)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]
    
    mean = tl.load(mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=offs_n < N, other=0.0)
    weight = tl.load(weight_ptr + offs_n, mask=offs_n < N, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    
    acc = (acc - mean) / tl.sqrt(var + eps)
    acc = acc * weight[None, :] + bias[None, :]
    
    gelu_out = 0.5 * acc * (1.0 + tl.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
    
    relu_out = tl.maximum(gelu_out, 0.0)
    
    out_ptrs = out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, relu_out, mask=mask)


def triton_gemm_bn_gelu_relu(x, w, b, mean, var, weight, bias, eps=1e-5):
    assert x.is_cuda and w.is_cuda and b.is_cuda and mean.is_cuda and var.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    
    gemm_bn_gelu_relu_kernel[grid](
        x, w, b, mean, var, weight, bias,
        out, None,
        M, K, N,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        w = self.gemm.weight
        b = self.gemm.bias
        mean = self.batch_norm.running_mean
        var = self.batch_norm.running_var
        weight = self.batch_norm.weight
        bias = self.batch_norm.bias
        return triton_gemm_bn_gelu_relu(x, w, b, mean, var, weight, bias, self.batch_norm.eps)