import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_features, out_features,
    stride_xb, stride_xi,
    stride_wb, stride_wo,
    stride_bb, stride_bo,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    mask_b = offs_b < batch_size
    mask_o = offs_o < out_features

    acc = tl.zeros([BLOCK_SIZE_B], dtype=tl.float32)

    for start_i in range(0, in_features, BLOCK_SIZE_O):
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_O)
        mask_i = offs_i < in_features

        x_idx = offs_b[:, None] * stride_xb + offs_i[None, :] * stride_xi
        w_idx = offs_o[None, :] * stride_wo + offs_i[:, None] * stride_wb

        x_blk = tl.load(x_ptr + x_idx, mask=mask_b[:, None] & mask_i[None, :], other=0.0)
        w_blk = tl.load(w_ptr + w_idx, mask=mask_o[None, :] & mask_i[:, None], other=0.0)

        acc += tl.sum(x_blk * w_blk, axis=1)

    if b_ptr is not None:
        b = tl.load(b_ptr + offs_o * stride_bb, mask=mask_o, other=0.0)
        acc += b

    # Sum reduction across out_features
    sum_val = tl.sum(acc)
    max_val = tl.max(acc)
    mean_val = sum_val / out_features

    # LogSumExp
    lse1 = tl.log(tl.sum(tl.exp(acc - max_val))) + max_val
    lse2 = tl.log(tl.sum(tl.exp(lse1 - max_val))) + max_val

    # Store final scalar result per batch
    tl.store(out_ptr + offs_b, lse2, mask=mask_b)


def triton_fused_forward(x, w, b):
    batch_size, in_features = x.shape
    out_features = w.shape[0]

    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    out = torch.empty(batch_size, dtype=torch.float32, device=x.device)

    BLOCK_SIZE_B = 32
    BLOCK_SIZE_O = 128

    grid = (
        (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B,
        (out_features + BLOCK_SIZE_O - 1) // BLOCK_SIZE_O,
    )

    fused_kernel[grid](
        x, w, b, out,
        batch_size, in_features, out_features,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0) if b is not None else 0, 1 if b is not None else 0,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_O=BLOCK_SIZE_O,
    )

    return out.unsqueeze(1)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return triton_fused_forward(x, self.linear.weight, self.linear.bias)