import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_matmul_kernel(
    x_ptr,  # Pointer to input tensor (batch_size, out_features)
    y_ptr,  # Pointer to output tensor (batch_size,)
    batch_size,
    out_features,
    pool_kernel_size,
    scale_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # batch_id
    offsets = tl.arange(0, BLOCK_SIZE)
    window_start = offsets * pool_kernel_size
    # Create pointers for loading the windows
    ptrs = x_ptr + pid * out_features + window_start[:, None] + tl.arange(0, pool_kernel_size)[None, :]
    vals = tl.load(ptrs)
    # Compute average
    avg = tl.sum(vals, axis=1) / pool_kernel_size
    # Scale
    scaled = avg * scale_factor
    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu_val = scaled * 0.5 * (1 + tl.erf(scaled / tl.sqrt(2.0)))
    # Reduce to max
    max_val = tl.reduce(gelu_val, 0, tl.maximum)
    # Store
    tl.store(y_ptr + pid, max_val)


def fused_post_matmul(x: torch.Tensor, scale_factor, pool_kernel_size):
    assert x.is_cuda and x.is_contiguous()
    batch_size, out_features = x.shape
    assert out_features % pool_kernel_size == 0
    y = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = out_features // pool_kernel_size
    grid = (batch_size,)
    fused_post_matmul_kernel[grid](
        x, y, batch_size, out_features, pool_kernel_size, scale_factor, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


class ModelNew(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max" with fused Triton kernel for post-matmul operations.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x = self.matmul(x)
        x = fused_post_matmul(x, self.scale_factor, self.pool_kernel_size)
        return x