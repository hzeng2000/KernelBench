import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    num_elements_per_batch,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements_per_batch
    ptr = x_ptr + batch_id * num_elements_per_batch + offsets
    x = tl.load(ptr, mask=mask, other=0.0)
    block_sum = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr + batch_id, block_sum)


def triton_mean(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    batch_size = x.shape[0]
    num_elements_per_batch = x.numel() // batch_size
    out = torch.zeros(batch_size, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 1024
    grid = (batch_size, (num_elements_per_batch + BLOCK_SIZE - 1) // BLOCK_SIZE)
    mean_kernel[grid](x, out, batch_size, num_elements_per_batch, BLOCK_SIZE=BLOCK_SIZE)
    out /= num_elements_per_batch
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = triton_mean(x)
        return x