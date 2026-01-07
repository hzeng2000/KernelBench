import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def clamp_mul_max_kernel(
    x_ptr, multiplier_ptr, out_ptr,
    batch, channels, depth, height, width,
    clamp_min, clamp_max
):
    pid = tl.program_id(0)
    spatial_size = depth * height * width
    b = pid // spatial_size
    spatial = pid % spatial_size
    d = spatial // (height * width)
    hw = spatial % (height * width)
    h = hw // width
    w = hw % width
    base_offset = b * (channels * spatial_size) + d * (height * width) + h * width + w
    offsets = base_offset + tl.arange(0, channels)
    x_vals = tl.load(x_ptr + offsets)
    mult_vals = tl.load(multiplier_ptr + tl.arange(0, channels))
    clamped = tl.clamp(x_vals, clamp_min, clamp_max)
    vals = clamped * mult_vals
    max_val = tl.max(vals)
    tl.store(out_ptr + pid, max_val)


def triton_clamp_mul_max(x: torch.Tensor, multiplier: torch.Tensor, clamp_min, clamp_max):
    assert x.is_cuda and multiplier.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    multiplier = multiplier.contiguous()
    batch, channels, depth, height, width = x.shape
    out = torch.empty(batch, depth, height, width, dtype=x.dtype, device=x.device)
    total_positions = batch * depth * height * width
    grid = (total_positions,)
    clamp_mul_max_kernel[grid](
        x, multiplier, out, batch, channels, depth, height, width, clamp_min, clamp_max
    )
    return out


class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = triton_clamp_mul_max(x, self.multiplier.squeeze(), self.clamp_min, self.clamp_max)
        return x