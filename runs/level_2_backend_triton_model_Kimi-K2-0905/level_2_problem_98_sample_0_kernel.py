import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_avgpool_gelu_scale_max_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_features, out_features,
    pool_kernel_size, scale_factor,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        mask_x = (rm[:, None] < batch_size) & (rk[None, :] < in_features)
        mask_w = (rk[:, None] < in_features) & (rn[None, :] < out_features)

        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk, mask=mask_x, other=0.0)
        w = tl.load(w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn, mask=mask_w, other=0.0)

        acc += tl.dot(x, w)

    if b_ptr is not None:
        b = tl.load(b_ptr + rn, mask=rn < out_features, other=0.0)
        acc += b[None, :]

    # AvgPool1d: kernel_size=pool_kernel_size, stride=pool_kernel_size
    # We reduce along the last dimension (out_features) in blocks of pool_kernel_size
    out_features_pooled = out_features // pool_kernel_size
    pooled = tl.zeros((BLOCK_SIZE_M, out_features_pooled), dtype=tl.float32)
    for i in range(out_features_pooled):
        start = i * pool_kernel_size
        end = start + pool_kernel_size
        pooled[:, i] = tl.sum(acc[:, start:end], axis=1) / pool_kernel_size

    # GELU activation
    gelu_out = 0.5 * pooled * (1.0 + tl.tanh(0.7978845608 * (pooled + 0.044715 * pooled * pooled * pooled)))

    # Scale
    scaled = gelu_out * scale_factor

    # Max along dim=1
    max_val = tl.max(scaled, axis=1)

    # Store output
    rm_valid = rm < batch_size
    tl.store(out_ptr + rm, max_val, mask=rm_valid)


def fused_matmul_avgpool_gelu_scale_max(x, weight, bias, pool_kernel_size, scale_factor):
    batch_size, in_features = x.shape
    out_features = weight.shape[0]

    # Allocate output tensor for final max values
    out = torch.empty(batch_size, dtype=torch.float32, device=x.device)

    # Grid dimensions
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: (
        triton.cdiv(batch_size, META["BLOCK_SIZE_M"]),
        triton.cdiv(out_features, META["BLOCK_SIZE_N"]),
    )

    matmul_avgpool_gelu_scale_max_kernel[grid](
        x, weight, bias, out,
        batch_size, in_features, out_features,
        pool_kernel_size, scale_factor,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), 1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        return fused_matmul_avgpool_gelu_scale_max(
            x, self.matmul.weight, self.matmul.bias,
            self.pool_kernel_size, self.scale_factor
        )