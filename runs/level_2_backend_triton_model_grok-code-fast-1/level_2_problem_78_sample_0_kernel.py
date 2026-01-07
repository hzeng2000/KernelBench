import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,
    out_channels,
    depth,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs for batch, depth, height, width
    b = tl.program_id(0)
    d = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    
    # Offsets for loading the channel dimension at this spatial position
    base_offset = b * (out_channels * depth * height * width) + d * (out_channels * height * width) + h * (out_channels * width) + w * out_channels
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < out_channels
    
    # Load the channel vector
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Sum over the channel dimension
    sum_val = tl.sum(x)
    
    # Output offset (keepdim=True, so dim=1 becomes 1)
    out_offset = b * (1 * depth * height * width) + 0 * (depth * height * width) + d * (height * width) + h * width + w
    tl.store(out_ptr + out_offset, sum_val)


def triton_sum(x: torch.Tensor):
    """
    Custom Triton-based sum over dim=1 (channels), keeping dim.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch, channels, d, h, w = x.shape
    out = torch.empty(batch, 1, d, h, w, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = channels  # Assuming channels fit in a block, e.g., 64
    
    grid = (batch, d, h, w)
    
    sum_kernel[grid](x, out, batch, channels, d, h, w, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by two max pooling layers and a custom Triton-based sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = triton_sum(x)
        return x