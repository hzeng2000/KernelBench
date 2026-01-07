import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def dropout_softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    p,
    scale,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * n_cols
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Generate random numbers for dropout
    rand = tl.rand(seed + tl.arange(0, BLOCK_SIZE).to(tl.uint32))
    drop_mask = rand < p
    x = tl.where(drop_mask, 0.0, x * scale)
    
    # Compute softmax
    max_x = tl.max(x)
    exp_x = tl.exp(x - max_x)
    sum_exp = tl.sum(exp_x)
    out = exp_x / sum_exp
    tl.store(output_ptr + offsets, out, mask=mask)


def triton_dropout_softmax(x: torch.Tensor, p: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    scale = 1.0 / (1.0 - p) if p < 1.0 else 1.0
    seed = torch.randint(0, 2**31, (1,), dtype=torch.int32).item()
    BLOCK_SIZE = n_cols
    grid = (n_rows,)
    dropout_softmax_kernel[grid](
        x, out, n_rows, n_cols, p, scale, seed, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    Optimized with Triton kernel for fused dropout and softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        if self.training:
            x = triton_dropout_softmax(x, self.dropout_p)
        else:
            x = torch.softmax(x, dim=1)
        return x