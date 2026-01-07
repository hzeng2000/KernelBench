import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_groupnorm_hardtanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_features, out_features,
    num_groups, eps,
    hardtanh_min, hardtanh_max,
    stride_xb, stride_xm,
    stride_wb, stride_wn,
    stride_outb, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_K):
        k_idx = k + rk
        mask_x = (rm[:, None] < batch_size) & (k_idx[None, :] < in_features)
        x_val = tl.load(x_ptr + rm[:, None] * stride_xm + k_idx[None, :] * stride_xb, mask=mask_x, other=0.0)
        mask_w = (rn[None, :] < out_features) & (k_idx[:, None] < in_features)
        w_val = tl.load(w_ptr + rn[None, :] * stride_wn + k_idx[:, None] * stride_wb, mask=mask_w, other=0.0)
        acc += tl.dot(x_val, w_val)

    if b_ptr is not None:
        b_val = tl.load(b_ptr + rn, mask=rn < out_features, other=0.0)
        acc += b_val[None, :]

    # GroupNorm
    group_size = out_features // num_groups
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        cols = start + tl.arange(0, BLOCK_N)
        mask_g = (rm[:, None] < batch_size) & (cols[None, :] >= start) & (cols[None, :] < end)
        vals = tl.where(mask_g, acc, 0.0)
        mean = tl.sum(vals, axis=1) / group_size
        var = tl.sum((vals - mean[:, None]) ** 2, axis=1) / group_size
        rstd = tl.rsqrt(var + eps)
        acc = tl.where(mask_g, (vals - mean[:, None]) * rstd[:, None], acc)

    # HardTanh
    acc = tl.where(acc < hardtanh_min, hardtanh_min, acc)
    acc = tl.where(acc > hardtanh_max, hardtanh_max, acc)

    # Store output
    mask_out = (rm[:, None] < batch_size) & (rn[None, :] < out_features)
    tl.store(out_ptr + rm[:, None] * stride_outn + rn[None, :] * stride_outb, acc, mask=mask_out)


def triton_gemm_groupnorm_hardtanh(x, weight, bias, num_groups, eps, hardtanh_min, hardtanh_max):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = ((batch_size + BLOCK_M - 1) // BLOCK_M, (out_features + BLOCK_N - 1) // BLOCK_N)

    gemm_groupnorm_hardtanh_kernel[grid](
        x, weight, bias, out,
        batch_size, in_features, out_features,
        num_groups, eps,
        hardtanh_min, hardtanh_max,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.eps = 1e-5

    def forward(self, x):
        return triton_gemm_groupnorm_hardtanh(
            x, self.gemm.weight, self.gemm.bias,
            self.num_groups, self.eps,
            self.hardtanh_min, self.hardtanh_max
        )