import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale(x: torch.Tensor, scale_factor: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    scale_kernel[grid](x, out, scale_factor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    partial_ptr,
    batch,
    channels,
    depth,
    height,
    width,
    num_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    block_id = tl.program_id(2)
    start = block_id * BLOCK_SIZE
    offsets_spatial = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets_spatial < num_spatial
    hw = height * width
    d_idx = offsets_spatial // hw
    temp = offsets_spatial % hw
    h_idx = temp // width
    w_idx = temp % width
    offset = (
        b * (channels * depth * height * width)
        + c * (depth * height * width)
        + d_idx * (height * width)
        + h_idx * width
        + w_idx
    )
    x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
    sum_val = tl.sum(x_val)
    partial_offset = b * channels + c
    tl.atomic_add(partial_ptr + partial_offset, sum_val)


def triton_global_avg_pool(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    batch, channels, depth, height, width = x.shape
    num_spatial = depth * height * width
    partial_sums = torch.zeros(batch, channels, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    num_blocks_per_channel = (num_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch, channels, num_blocks_per_channel)
    global_avg_pool_kernel[grid](
        x, partial_sums, batch, channels, depth, height, width, num_spatial, BLOCK_SIZE=BLOCK_SIZE
    )
    out = partial_sums / num_spatial
    return out.view(batch, channels, 1, 1, 1)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_scale(x, self.scale_factor)
        x = self.batch_norm(x)
        x = triton_global_avg_pool(x)
        return x


batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]