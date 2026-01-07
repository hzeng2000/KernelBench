import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_groupnorm_min_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, bias_ptr,
    M, N, K,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_idx = k * BLOCK_K + rk
        x_idx = x_ptr + (rm[:, None] * K + k_idx[None, :])
        w_idx = w_ptr + (k_idx[:, None] * N + rn[None, :])
        mask_x = (rm[:, None] < M) & (k_idx[None, :] < K)
        mask_w = (k_idx[:, None] < K) & (rn[None, :] < N)
        x_block = tl.load(x_idx, mask=mask_x, other=0.0)
        w_block = tl.load(w_idx, mask=mask_w, other=0.0)
        acc += tl.dot(x_block, w_block)

    b_block = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    acc = acc + b_block[None, :]

    # GroupNorm: compute mean and var per group
    group_size = N // 512
    group = pid_n * BLOCK_N // group_size
    acc_flat = acc.to(tl.float32)
    mean = tl.sum(acc_flat, axis=1) / N
    var = tl.sum((acc_flat - mean[:, None]) ** 2, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    acc = (acc_flat - mean[:, None]) * rstd[:, None]

    # Min along dim=1
    min_val = tl.min(acc, axis=1)
    min_out = min_val[:, None]

    # Add bias
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
    out = min_out + bias[None, :]

    out_idx = out_ptr + (rm[:, None] * N + rn[None, :])
    mask_out = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_idx, out, mask=mask_out)


def fused_gemm_groupnorm_min_bias(x, w, b, bias, num_groups=512):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    gemm_groupnorm_min_bias_kernel[grid](
        x, w, b, out, bias,
        M, N, K,
        eps=1e-5,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.num_groups = num_groups
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        w = self.gemm.weight
        b = self.gemm.bias
        x = fused_gemm_groupnorm_min_bias(x, w, b, self.bias.view(-1), self.num_groups)
        return x.unsqueeze(-1).unsqueeze(-1)