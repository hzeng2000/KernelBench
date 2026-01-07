import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, input_size, hidden_size,
    stride_xb, stride_xm,
    stride_wh, stride_wn,
    stride_ob, stride_om,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    batch_offs = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    hidden_offs = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    batch_mask = batch_offs < batch_size
    hidden_mask = hidden_offs < hidden_size

    acc = tl.zeros([BLOCK_SIZE_B], dtype=tl.float32)

    for k_start in range(0, input_size, BLOCK_SIZE_H):
        k_offs = k_start + tl.arange(0, BLOCK_SIZE_H)
        k_mask = k_offs < input_size

        x_idx = (batch_offs[:, None] * stride_xb + k_offs[None, :] * stride_xm)
        w_idx = (k_offs[:, None] * stride_wh + hidden_offs[None, :] * stride_wn)

        x_blk = tl.load(x_ptr + x_idx, mask=batch_mask[:, None] & k_mask[None, :], other=0.0)
        w_blk = tl.load(w_ptr + w_idx, mask=k_mask[:, None] & hidden_mask[None, :], other=0.0)

        acc += tl.sum(x_blk[:, :, None] * w_blk[None, :, :], axis=1)

    b_vec = tl.load(b_ptr + hidden_offs, mask=hidden_mask, other=0.0)
    acc = acc + b_vec

    sigmoid_out = tl.sigmoid(acc)

    out_idx = batch_offs * stride_ob + pid_h
    tl.store(out_ptr + out_idx, sigmoid_out, mask=batch_mask)

    if pid_h == 0:
        final_acc = tl.zeros([BLOCK_SIZE_B], dtype=tl.float32)
        for h in range(0, hidden_size, BLOCK_SIZE_H):
            h_offs = h + tl.arange(0, BLOCK_SIZE_H)
            h_mask = h_offs < hidden_size
            out_idx = batch_offs * stride_ob + h_offs
            vals = tl.load(out_ptr + out_idx, mask=batch_mask[:, None] & h_mask[None, :], other=0.0)
            final_acc += tl.sum(vals, axis=1)
        final_idx = batch_offs
        tl.store(out_ptr + final_idx, final_acc, mask=batch_mask)


def triton_linear_sigmoid_sum(x, weight, bias):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    batch_size, input_size = x.shape
    hidden_size = weight.shape[0]

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty(batch_size, hidden_size, dtype=torch.float32, device=x.device)

    BLOCK_SIZE_B = 32
    BLOCK_SIZE_H = 128

    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE_B"] - 1) // meta["BLOCK_SIZE_B"],
                         (hidden_size + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"])

    linear_sigmoid_sum_kernel[grid](
        x, weight, bias, out,
        batch_size, input_size, hidden_size,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H
    )

    final_out = torch.empty(batch_size, 1, dtype=torch.float32, device=x.device)
    for b in range(batch_size):
        final_out[b, 0] = out[b, :hidden_size].sum()

    return final_out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return triton_linear_sigmoid_sum(x, self.linear.weight, self.linear.bias)