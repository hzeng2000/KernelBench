import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale_factor, clamp_min, clamp_max,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Load bias
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    accumulator += bias[None, :]

    # Scale and residual (*2)
    accumulator = accumulator * (2 * scale_factor)

    # Clamp
    accumulator = tl.clamp(accumulator, clamp_min, clamp_max)

    tl.store(c_ptrs, accumulator, mask=mask)


def triton_fused_matmul(a, b, bias, scale_factor, clamp_min, clamp_max):
    assert a.is_cuda and b.is_cuda and bias.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    fused_matmul_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scale_factor, clamp_min, clamp_max,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
    )
    return c


@triton.jit
def logsumexp_mish_kernel(
    x_ptr, out_ptr,
    batch_size, hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    max_val = -float('inf')
    for start in range(0, hidden_size, BLOCK_SIZE):
        offsets = pid * hidden_size + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (pid + 1) * hidden_size
        x_block = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_block))
    sum_exp = 0.0
    for start in range(0, hidden_size, BLOCK_SIZE):
        offsets = pid * hidden_size + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (pid + 1) * hidden_size
        x_block = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(x_block - max_val))
    logsum = max_val + tl.log(sum_exp)
    softplus = tl.log(1 + tl.exp(logsum))
    tanh_softplus = tl.tanh(softplus)
    mish = logsum * tanh_softplus
    tl.store(out_ptr + pid, mish)


def triton_logsumexp_mish(x):
    assert x.is_cuda
    x = x.contiguous()
    batch_size, hidden_size = x.shape
    out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
    grid = (batch_size,)
    BLOCK_SIZE = 1024  # Adjust based on hidden_size
    logsumexp_mish_kernel[grid](x, out, batch_size, hidden_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = triton_fused_matmul(x, self.matmul.weight.t(), self.matmul.bias, self.scale_factor, self.clamp_min, self.clamp_max)
        x = triton_logsumexp_mish(x)
        return x