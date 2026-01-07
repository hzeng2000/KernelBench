import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    min_val,  # Minimum value for min and clamp
    max_val,  # Maximum value for clamp
    dropout_p,  # Dropout probability
    seed,  # Random seed for dropout
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply torch.min(x, min_value)
    x = tl.minimum(x, min_val)
    # Apply torch.clamp(x, min=min_value, max=max_value)
    x = tl.clamp(x, min_val, max_val)
    # Apply dropout
    rand_val = tl.rand(seed, offsets)
    drop_mask = rand_val > dropout_p
    scale = 1.0 / (1.0 - dropout_p)
    x = tl.where(drop_mask, x * scale, 0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_post(x: torch.Tensor, min_val: float, max_val: float, dropout_p: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    seed = tl.randint(0, 2**31 - 1)  # Generate a random seed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_post_kernel[grid](x, out, min_val, max_val, dropout_p, seed, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, and then fused minimum, clamp, and dropout using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = fused_post(x, self.min_value, self.max_value, self.dropout_p)
        return x