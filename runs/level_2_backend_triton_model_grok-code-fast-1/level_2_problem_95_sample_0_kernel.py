import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_activations_kernel(
    a_ptr,  # x: (M, K)
    b_ptr,  # weight.T: (K, N)
    bias_ptr,  # combined_bias: (N,)
    c_ptr,  # output: (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M and offs_k[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K and offs_bn[None, :] < N, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator

    # Add bias
    bias_vals = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    c += bias_vals[None, :]

    # Swish: sigmoid(c) * c
    sig = tl.sigmoid(c)
    c = sig * c

    # Tanh
    c = tl.tanh(c)

    # GELU
    sqrt_2_pi = tl.sqrt(2.0 / tl.pi)
    inner = sqrt_2_pi * (c + 0.044715 * c * c * c)
    c = 0.5 * c * (1.0 + tl.tanh(inner))

    # Hardtanh
    c = tl.clamp(c, -1.0, 1.0)

    offs_cm = offs_am
    offs_cn = offs_bn
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=offs_cm[:, None] < M and offs_cn[None, :] < N)


def triton_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight_t = weight.T.contiguous()
    bias = bias.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    fused_matmul_activations_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        combined_bias = self.bias + self.add_value
        return triton_forward(x, self.weight, combined_bias)