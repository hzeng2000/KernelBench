import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_scale_kernel(
    x_ptr, weight_ptr, bias_ptr, scale_ptr, out_ptr,
    batch_size, in_features, out_features,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_features, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * in_features + offs_k[None, :])
    weight_ptrs = weight_ptr + (offs_k[:, None] * out_features + offs_n[None, :])
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < batch_size and offs_k[None, :] < in_features, other=0.0)
        weight = tl.load(weight_ptrs, mask=offs_k[:, None] < in_features and offs_n[None, :] < out_features, other=0.0)
        accumulator = tl.dot(x, weight, accumulator)
        x_ptrs += BLOCK_SIZE_K
        weight_ptrs += BLOCK_SIZE_K * out_features
    offs_bias = offs_n
    bias = tl.load(bias_ptr + offs_bias, mask=offs_n < out_features, other=0.0)
    accumulator += bias[None, :]
    scale = tl.load(scale_ptr + offs_bias, mask=offs_n < out_features, other=0.0)
    accumulator *= scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * out_features + offs_n[None, :]
    tl.store(out_ptrs, accumulator, mask=offs_m[:, None] < batch_size and offs_n[None, :] < out_features)


def triton_fused_linear_scale(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and scale.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    scale = scale.contiguous()
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    out = torch.empty((batch_size, out_features), dtype=torch.float32, device=x.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE_M']) * triton.cdiv(out_features, meta['BLOCK_SIZE_N']), )
    fused_linear_scale_kernel[grid](
        x, weight, bias, scale, out,
        batch_size, in_features, out_features,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = triton_fused_linear_scale(x, self.gemm.weight, self.gemm.bias, self.scale)
        x = self.bn(x)
        return x