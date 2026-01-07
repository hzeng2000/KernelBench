import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_sum_kernel(
    x_ptr,  # (batch, in_f)
    w_ptr,  # (out_f, in_f)
    b_ptr,  # (out_f,)
    out_ptr,  # (batch, 1)
    batch, in_f, out_f,
    BLOCK_IN: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid
    if batch_idx >= batch:
        return
    # Offsets for x: x[batch_idx, :]
    x_offsets = batch_idx * in_f + tl.arange(0, BLOCK_IN)
    # Accumulator
    acc = 0.0
    for out_start in range(0, out_f, BLOCK_OUT):
        out_offsets = out_start + tl.arange(0, BLOCK_OUT)
        mask_out = out_offsets < out_f
        # Load b
        b = tl.load(b_ptr + out_offsets, mask=mask_out, other=0.0)
        # Load w block: (BLOCK_OUT, BLOCK_IN)
        w_block_ptr = w_ptr + out_offsets[:, None] * in_f + tl.arange(0, BLOCK_IN)[None, :]
        w = tl.load(w_block_ptr, mask=mask_out[:, None], other=0.0)
        # Load x block
        x_block = tl.load(x_ptr + x_offsets, mask=tl.arange(0, BLOCK_IN) < in_f, other=0.0)
        # Compute dot: w @ x for each out in block
        dot = tl.sum(w * x_block[None, :], axis=1)
        dot += b
        # Accumulate sum over BLOCK_OUT
        acc += tl.sum(dot)
    # Store to out[batch_idx, 0]
    tl.store(out_ptr + batch_idx, acc)


def fused_linear_sum(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    batch, in_f = x.shape
    out_f = w.shape[0]
    out = torch.empty(batch, 1, device=x.device, dtype=x.dtype)
    BLOCK_IN = 128
    BLOCK_OUT = 32
    grid = (batch,)
    fused_linear_sum_kernel[grid](
        x, w, b, out, batch, in_f, out_f, BLOCK_IN=BLOCK_IN, BLOCK_OUT=BLOCK_OUT
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = fused_linear_sum(x, self.linear.weight, self.linear.bias)  # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0]  # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True)  # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True)  # (batch_size, 1)
        return x