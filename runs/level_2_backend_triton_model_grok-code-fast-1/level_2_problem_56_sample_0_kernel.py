import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_sigmoid_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch,
    input_size,
    hidden_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    n_block_start = pid_n * BLOCK_SIZE_N
    n_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < hidden_size

    m_block_start = pid_m * BLOCK_SIZE_M
    m_offsets = m_block_start + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < batch

    k_block_start = pid_k * BLOCK_SIZE_K
    k_offsets = k_block_start + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < input_size

    a_ptrs = x_ptr + m_offsets[:, None] * input_size + k_offsets[None, :]
    b_ptrs = weight_ptr + n_offsets[:, None] * input_size + k_offsets[None, :]

    a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

    c = tl.dot(a, b)

    bias_ptrs = bias_ptr + n_offsets
    bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
    c += bias_vals[None, :]

    c = tl.sigmoid(c)

    out_ptrs = out_ptr + m_offsets[:, None] * hidden_size + n_offsets[None, :]
    tl.store(out_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def triton_linear_sigmoid(x, weight, bias):
    batch, input_size = x.shape
    hidden_size = weight.shape[0]
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty(batch, hidden_size, dtype=x.dtype, device=x.device)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    num_pid_m = (batch + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_k = (input_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid = (num_pid_n, num_pid_m, num_pid_k)
    linear_sigmoid_kernel[grid](
        x, weight, bias, out, batch, input_size, hidden_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out


@triton.jit
def sum_kernel(
    x_ptr,
    out_ptr,
    batch,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch:
        return
    sum_val = 0.0
    for start in tl.range(0, hidden_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        vals = tl.load(x_ptr + pid * hidden_size + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    tl.store(out_ptr + pid, sum_val)


def triton_sum(x):
    batch, hidden_size = x.shape
    out = torch.empty(batch, 1, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = (batch,)
    sum_kernel[grid](x, out, batch, hidden_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = triton_linear_sigmoid(x, self.weight, self.bias)
        x = triton_sum(x)
        return x