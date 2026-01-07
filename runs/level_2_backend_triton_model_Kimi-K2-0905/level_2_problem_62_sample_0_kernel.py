import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_gn_leakyrelu_add_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, input_size, hidden_size,
    group_size, eps, negative_slope,
    stride_xb, stride_xm,
    stride_wb, stride_wn,
    stride_outb, stride_outn,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask_b = offs_b < batch_size
    mask_h = offs_h < hidden_size

    # Compute group mean and variance
    group_idx = offs_h // group_size
    group_start = group_idx * group_size
    group_end = (group_idx + 1) * group_size

    # Compute matmul: out = x @ w.T + b
    acc = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_H], dtype=tl.float32)
    for k in range(0, input_size, 32):
        k_offs = k + tl.arange(0, 32)
        mask_k = k_offs < input_size
        x_vals = tl.load(x_ptr + offs_b[:, None] * stride_xb + k_offs[None, :] * stride_xm, mask=mask_b[:, None] & mask_k[None, :], other=0.0)
        w_vals = tl.load(w_ptr + offs_h[None, :] * stride_wb + k_offs[:, None] * stride_wn, mask=mask_h[None, :] & mask_k[:, None], other=0.0)
        acc += tl.dot(x_vals, w_vals)

    b_vals = tl.load(b_ptr + offs_h, mask=mask_h, other=0.0)
    out = acc + b_vals[None, :]

    # GroupNorm within block
    mean = tl.sum(out, axis=1) / hidden_size
    var = tl.sum((out - mean[:, None]) ** 2, axis=1) / hidden_size
    std = tl.sqrt(var + eps)
    out_norm = (out - mean[:, None]) / std[:, None]

    # LeakyReLU
    out_relu = tl.where(out_norm > 0, out_norm, out_norm * negative_slope)

    # Element-wise add (x + x)
    out_final = out_relu + out_relu

    tl.store(out_ptr + offs_b[:, None] * stride_outb + offs_h[None, :] * stride_outn, out_final, mask=mask_b[:, None] & mask_h[None, :])


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.eps = eps
        self.negative_slope = negative_slope
        self.num_groups = num_groups

    def forward(self, x):
        batch_size, input_size = x.shape
        hidden_size = self.fc.out_features
        group_size = hidden_size // self.num_groups

        w = self.fc.weight
        b = self.fc.bias

        out = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)

        BLOCK_SIZE_B = 32
        BLOCK_SIZE_H = 128

        grid = ((batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B,
                (hidden_size + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H)

        fused_matmul_gn_leakyrelu_add_kernel[grid](
            x, w, b, out,
            batch_size, input_size, hidden_size,
            group_size, self.eps, self.negative_slope,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )

        return out