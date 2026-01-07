import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_maxpool_sum_scale_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_features, out_features,
    kernel_size, scale_factor,
    stride_x, stride_w, stride_b, stride_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    rm_mask = rm < batch_size
    rn_mask = rn < out_features
    rk_mask = rk < in_features

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, in_features, BLOCK_SIZE_K):
        k_idx = k + rk
        x_idx = rm[:, None] * stride_x + k_idx[None, :]
        w_idx = k_idx[:, None] * stride_w + rn[None, :]

        x_mask = rm_mask[:, None] & rk_mask[None, :]
        w_mask = rk_mask[:, None] & rn_mask[None, :]

        x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
        w_val = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc += tl.dot(x_val, w_val)

    if b_ptr is not None:
        b_idx = rn
        b_mask = rn_mask
        b_val = tl.load(b_ptr + b_idx, mask=b_mask, other=0.0)
        acc += b_val[None, :]

    # MaxPool1d with kernel_size=2
    out_idx = rm[:, None] * stride_out + rn[None, :]
    out_mask = rm_mask[:, None] & rn_mask[None, :]

    # MaxPool1d: take max of adjacent pairs
    pooled = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), float('-inf'), dtype=tl.float32)
    for i in range(0, BLOCK_SIZE_N, 2):
        if i + 1 < BLOCK_SIZE_N:
            pooled[:, i // 2] = tl.maximum(acc[:, i], acc[:, i + 1])
        else:
            pooled[:, i // 2] = acc[:, i]

    # Sum over dim=1
    sum_val = tl.sum(pooled, axis=1)

    # Scale
    scaled = sum_val * scale_factor

    # Store final result
    final_idx = rm
    final_mask = rm_mask
    tl.store(out_ptr + final_idx, scaled, mask=final_mask)


def triton_fused_forward(x, weight, bias, kernel_size, scale_factor):
    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty(batch_size, dtype=torch.float32, device=x.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = lambda META: (
        (batch_size + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"],
        (out_features + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"],
    )

    fused_matmul_maxpool_sum_scale_kernel[grid](
        x, weight, bias, out,
        batch_size, in_features, out_features,
        kernel_size, scale_factor,
        x.stride(0), weight.stride(0), bias.stride(0) if bias is not None else 0, out.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        return triton_fused_forward(x, self.matmul.weight, self.matmul.bias, self.kernel_size, self.scale_factor)