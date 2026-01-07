import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_min_kernel(
    x_ptr,
    y_ptr,
    scale_factor,
    batch,
    out_channels,
    height,
    width,
):
    pid = tl.program_id(0)
    total_spatial = height * width
    b = pid // total_spatial
    hw = pid % total_spatial
    h = hw // width
    w = hw % width

    # Base offset for this b, h, w
    base_offset = b * (out_channels * height * width) + h * width + w
    stride = height * width

    # Offsets for all channels
    offsets = base_offset + tl.arange(0, out_channels) * stride

    # Load the vector for all channels
    vec = tl.load(x_ptr + offsets)

    # Scale
    vec = vec * scale_factor

    # Reduce to min
    min_val = tl.reduce(vec, 0, tl.minimum)

    # Store to output
    y_offset = b * (1 * height * width) + h * width + w
    tl.store(y_ptr + y_offset, min_val)


def triton_scale_min(x: torch.Tensor, scale_factor: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    batch, out_channels, height, width = x.shape
    y = torch.empty(batch, 1, height, width, device=x.device, dtype=x.dtype)
    grid = (batch * height * width,)
    scale_min_kernel[grid](x, y, scale_factor, batch, out_channels, height, width)
    return y


class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = triton_scale_min(x, self.scale_factor)
        return x