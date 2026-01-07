import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_mul_leakyrelu_kernel(
    x_ptr,  # input: (M, K)
    w_ptr,  # weight: (N, K)
    b_ptr,  # bias: (N,)
    out_ptr,  # output: (M, N)
    multiplier,  # scalar
    negative_slope,  # for leakyrelu
    M, N, K,  # dimensions
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = min(BLOCK_SIZE_K, K - k * BLOCK_SIZE_K)
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
    # Load bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    # Add bias
    accumulator += b[None, :]
    # Multiply by scalar
    accumulator *= multiplier
    # Apply leaky relu
    accumulator = tl.where(accumulator > 0, accumulator, accumulator * negative_slope)
    # Store
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_gemm_mul_leakyrelu(x, w, b, multiplier, negative_slope):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    M, K = x.shape
    N, K_ = w.shape
    assert K == K_
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    fused_gemm_mul_leakyrelu_kernel[grid](
        x, w, b, out, multiplier, negative_slope,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses Gemm, multiplication, and LeakyReLU into a single Triton kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        # Initialize weights and bias as in nn.Linear
        self.w = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.randn(out_features))
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        return fused_gemm_mul_leakyrelu(x, self.w, self.b, self.multiplier, self.negative_slope)