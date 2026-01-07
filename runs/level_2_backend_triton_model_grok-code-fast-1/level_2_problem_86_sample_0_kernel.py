import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_linear_div_gelu_kernel(
    x_ptr,  # Input tensor: (M, K)
    w_ptr,  # Weight tensor: (N, K)
    b_ptr,  # Bias tensor: (N,)
    out_ptr,  # Output tensor: (M, N)
    divisor,  # Scalar divisor
    M,  # Batch size
    N,  # Output size
    K,  # Input size
    stride_xm, stride_xk,  # Strides for x
    stride_wk, stride_wn,  # Strides for w (note: w is (N, K), so stride_wn is K, stride_wk is 1)
    stride_b,  # Stride for b (1)
    stride_om, stride_on,  # Strides for out
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
    b_ptrs = b_ptr + offs_n * stride_b
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        mask_k_curr = mask_k & (k_offs < K)
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k_curr[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k_curr[None, :], other=0.0)
        accumulator += tl.dot(x, w, trans_b=True)  # Note: trans_b=True since w is (N, K) and we want x @ w.T
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # Add bias
    b = tl.load(b_ptrs, mask=mask_n, other=0.0)
    accumulator += b[None, :]

    # Divide by divisor
    accumulator = accumulator / divisor

    # Apply GELU
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = math.sqrt(2 / math.pi)
    x_gelu = accumulator
    x_cubed = x_gelu * x_gelu * x_gelu
    inner = sqrt_2_pi * (x_gelu + 0.044715 * x_cubed)
    tanh_inner = tl.extra.cuda.libdevice.tanh(inner)
    gelu_out = 0.5 * x_gelu * (1 + tanh_inner)

    # Store
    tl.store(out_ptrs, gelu_out, mask=mask_m[:, None] & mask_n[None, :])


def fused_linear_div_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, divisor: float):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, "Input size mismatch"
    assert bias.shape == (N,), "Bias shape mismatch"

    out = torch.empty((M, N), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    fused_linear_div_gelu_kernel[grid](
        x, weight, bias, out, divisor,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # stride_wk=1, stride_wn=K
        bias.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return out


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    Optimized with a fused Triton kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Use the fused Triton kernel instead of separate operations
        return fused_linear_div_gelu(x, self.linear.weight, self.linear.bias, self.divisor)