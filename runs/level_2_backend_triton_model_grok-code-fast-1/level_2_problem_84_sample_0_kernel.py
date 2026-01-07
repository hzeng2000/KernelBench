import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K, Z,
    stride_z_a, stride_m_a, stride_k_a,
    stride_z_b, stride_k_b, stride_n_b,
    stride_z_c, stride_m_c, stride_n_c,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_z = tl.program_id(2)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + pid_z * stride_z_a + offs_m[:, None] * stride_m_a + offs_k[None, :] * stride_k_a
    b_ptrs = b_ptr + pid_z * stride_z_b + offs_k[:, None] * stride_k_b + offs_n[None, :] * stride_n_b
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_k_a
        b_ptrs += BLOCK_SIZE_K * stride_k_b
    c = accumulator
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    c += bias[None, :]
    offs_m_mask = offs_m[:, None] < M
    offs_n_mask = offs_n[None, :] < N
    mask = offs_m_mask & offs_n_mask
    tl.store(c_ptr + pid_z * stride_z_c + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_n_c, c, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    assert a.is_cuda and b.is_cuda and bias.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    batch_size, in_features = a.shape
    _, out_features = b.shape
    out = torch.empty(batch_size, out_features, device=a.device, dtype=a.dtype)
    M = 1
    N = out_features
    K = in_features
    Z = batch_size
    stride_z_a = in_features
    stride_m_a = in_features
    stride_k_a = 1
    stride_z_b = 0
    stride_k_b = out_features
    stride_n_b = 1
    stride_z_c = out_features
    stride_m_c = out_features
    stride_n_c = 1
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 1024
    BLOCK_SIZE_K = 128
    grid = ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, Z)
    matmul_kernel[grid](
        a, b, out, bias,
        M, N, K, Z,
        stride_z_a, stride_m_a, stride_k_a,
        stride_z_b, stride_k_b, stride_n_b,
        stride_z_c, stride_m_c, stride_n_c,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


@triton.jit
def bn_stats_kernel(
    c_ptr, mean_ptr, var_ptr, running_mean_ptr, running_var_ptr,
    batch, out_feat,
    stride_z_c, stride_m_c, stride_n_c,
    eps, momentum,
    training: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid
    if n >= out_feat:
        return
    offsets_z = tl.arange(0, batch)
    vals = tl.load(c_ptr + offsets_z * stride_z_c + 0 * stride_m_c + n * stride_n_c)
    mean = tl.sum(vals) / batch
    var = tl.sum((vals - mean) ** 2) / (batch - 1)
    tl.store(mean_ptr + n, mean)
    tl.store(var_ptr + n, var)
    if training:
        running_mean = tl.load(running_mean_ptr + n)
        running_var = tl.load(running_var_ptr + n)
        new_running_mean = momentum * running_mean + (1 - momentum) * mean
        new_running_var = momentum * running_var + (1 - momentum) * var
        tl.store(running_mean_ptr + n, new_running_mean)
        tl.store(running_var_ptr + n, new_running_var)


@triton.jit
def fused_bn_scale_softmax_kernel(
    c_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, scale,
    batch, out_feat,
    stride_z_c, stride_m_c, stride_n_c,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    z = pid
    if z >= batch:
        return
    offsets_n = tl.arange(0, BLOCK_SIZE)
    mask = offsets_n < out_feat
    vals = tl.load(c_ptr + z * stride_z_c + 0 * stride_m_c + offsets_n * stride_n_c, mask=mask)
    mean = tl.load(mean_ptr + offsets_n, mask=mask)
    var = tl.load(var_ptr + offsets_n, mask=mask)
    weight = tl.load(weight_ptr + offsets_n, mask=mask)
    bn_bias = tl.load(bias_ptr + offsets_n, mask=mask)
    normalized = (vals - mean) / tl.sqrt(var + eps) * weight + bn_bias
    scaled = normalized * scale
    max_val = tl.max(scaled)
    exp_vals = tl.exp(scaled - max_val)
    sum_exp = tl.sum(exp_vals)
    softmax_vals = exp_vals / sum_exp
    tl.store(c_ptr + z * stride_z_c + 0 * stride_m_c + offsets_n * stride_n_c, softmax_vals, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.bn_running_mean = torch.zeros(out_features)
        self.bn_running_var = torch.ones(out_features)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.out_features = out_features

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size = x.shape[0]
        x = triton_matmul(x, self.gemm_weight, self.gemm_bias)
        mean = torch.empty(self.out_features, device=x.device, dtype=x.dtype)
        var = torch.empty(self.out_features, device=x.device, dtype=x.dtype)
        bn_stats_kernel[(self.out_features,)](
            x, mean, var, self.bn_running_mean, self.bn_running_var,
            batch_size, self.out_features,
            self.out_features, self.out_features, 1,
            self.bn_eps, self.bn_momentum,
            self.training
        )
        fused_bn_scale_softmax_kernel[(batch_size,)](
            x, mean, var, self.bn_weight, self.bn_bias, self.scale.item(),
            batch_size, self.out_features,
            self.out_features, self.out_features, 1,
            self.bn_eps,
            self.out_features
        )
        return x