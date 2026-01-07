import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_f, out_f,
    stride_xb, stride_xi,
    stride_wi, stride_wo,
    stride_ob, stride_oo,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + offs_m[:, None, None] * stride_xb + offs_k[None, :, None] * stride_xi
    w_ptrs = w_ptr + offs_k[None, :, None] * stride_wi + offs_n[None, None, :] * stride_wo
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, in_f, BLOCK_SIZE_K):
        x_mask = (offs_m[:, None, None] < batch) & (offs_k[None, :, None] < in_f)
        w_mask = (offs_k[None, :, None] < in_f) & (offs_n[None, None, :] < out_f)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_xi
        w_ptrs += BLOCK_SIZE_K * stride_wi
    b_ptrs = b_ptr + offs_n
    b = tl.load(b_ptrs, mask=offs_n < out_f, other=0.0)
    accumulator += b[None, :]
    out_mask = (offs_m[:, None] < batch) & (offs_n[None, :] < out_f)
    tl.store(out_ptr + offs_m[:, None] * stride_ob + offs_n[None, :] * stride_oo, accumulator, mask=out_mask)


def triton_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    batch, in_f = x.shape
    out_f = w.shape[0]
    out = torch.empty(batch, out_f, device=x.device, dtype=torch.float32)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    grid = ((batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (out_f + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    linear_kernel[grid](
        x, w, b, out, batch, in_f, out_f,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


@triton.jit
def min_sub_kernel(
    x_ptr, out_ptr, constant, n_elements, BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.minimum(x, constant) - constant
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_min_sub(x: torch.Tensor, constant: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    min_sub_kernel[grid](x, out, constant, n_elements, BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for linear and fused min-subtract operations.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        x = triton_linear(x, self.linear.weight, self.linear.bias)
        x = triton_min_sub(x, self.constant.item())
        return x