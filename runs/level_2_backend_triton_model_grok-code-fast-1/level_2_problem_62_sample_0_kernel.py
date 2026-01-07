import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr,  # Pointer to input x: (batch_size, input_size)
    w_ptr,  # Pointer to weight: (hidden_size, input_size)
    b_ptr,  # Pointer to bias: (hidden_size,)
    out_ptr,  # Pointer to output: (batch_size, hidden_size)
    batch_size,
    input_size,
    hidden_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers
    x_ptrs = x_ptr + offs_m[:, None] * input_size + offs_k[None, :]
    w_ptrs = w_ptr + offs_k[:, None] * hidden_size + offs_n[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * hidden_size + offs_n[None, :]

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, input_size, BLOCK_SIZE_K):
        x_mask = offs_m[:, None] < batch_size
        w_mask = offs_n[None, :] < hidden_size
        k_mask = offs_k < input_size - k

        x = tl.load(x_ptrs, mask=x_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & w_mask[None, :], other=0.0)
        acc += tl.dot(x, w)

        x_ptrs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K * hidden_size

    # Load bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < hidden_size, other=0.0)
    acc += b[None, :]

    # Store
    out_mask = (offs_m[:, None] < batch_size) & (offs_n[None, :] < hidden_size)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, input_size = x.shape
    hidden_size = weight.shape[0]

    out = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    grid = (
        (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )

    linear_kernel[grid](
        x, weight, bias, out,
        batch_size, input_size, hidden_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


@triton.jit
def leaky_double_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, negative_slope * x) * 2.0
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_leaky_double(x: torch.Tensor, negative_slope: float):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    leaky_double_kernel[grid](x, out, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = triton_linear(x, self.fc.weight, self.fc.bias)
        x = self.gn(x)
        x = triton_leaky_double(x, self.negative_slope)
        return x