import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def compute_local_max_kernel(
    x_ptr,
    temp_max_ptr,
    b, c, d, h, w,
    BLOCK_SIZE: tl.constexpr,
    num_blocks: tl.constexpr,
):
    pid = tl.program_id(0)
    group_id = pid // num_blocks
    block_id = pid % num_blocks
    batch = group_id // c
    chan = group_id % c
    N = d * h * w
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < N
    spatial_offsets = block_start + offsets
    base_offset = batch * (c * d * h * w) + chan * (d * h * w)
    x_offsets = base_offset + spatial_offsets
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=-float('inf'))
    local_max = tl.max(x_vals)
    temp_max_offset = group_id * num_blocks + block_id
    tl.store(temp_max_ptr + temp_max_offset, local_max)

@triton.jit
def reduce_max_kernel(
    temp_max_ptr,
    global_max_ptr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    vals = tl.load(temp_max_ptr + group_id * num_blocks + offsets, mask=mask, other=-float('inf'))
    global_max = tl.max(vals)
    tl.store(global_max_ptr + group_id, global_max)

@triton.jit
def compute_local_sum_kernel(
    x_ptr,
    temp_sum_ptr,
    global_max_ptr,
    b, c, d, h, w,
    BLOCK_SIZE: tl.constexpr,
    num_blocks: tl.constexpr,
):
    pid = tl.program_id(0)
    group_id = pid // num_blocks
    block_id = pid % num_blocks
    batch = group_id // c
    chan = group_id % c
    N = d * h * w
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < N
    spatial_offsets = block_start + offsets
    base_offset = batch * (c * d * h * w) + chan * (d * h * w)
    x_offsets = base_offset + spatial_offsets
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    global_max = tl.load(global_max_ptr + group_id)
    exp_vals = tl.exp(x_vals - global_max)
    local_sum = tl.sum(exp_vals)
    temp_sum_offset = group_id * num_blocks + block_id
    tl.store(temp_sum_ptr + temp_sum_offset, local_sum)

@triton.jit
def reduce_sum_kernel(
    temp_sum_ptr,
    global_sum_ptr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    vals = tl.load(temp_sum_ptr + group_id * num_blocks + offsets, mask=mask, other=0.0)
    global_sum = tl.sum(vals)
    tl.store(global_sum_ptr + group_id, global_sum)

@triton.jit
def compute_softmax_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    global_max_ptr,
    global_sum_ptr,
    clamp_min,
    clamp_max,
    b, c, d, h, w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = b * c * d * h * w
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    x_vals = tl.clamp(x_vals, clamp_min, clamp_max)
    N = d * h * w
    group_id = offsets // N
    global_max = tl.load(global_max_ptr + group_id, mask=mask)
    global_sum = tl.load(global_sum_ptr + group_id, mask=mask)
    chan = (offsets // N) % c
    scale_val = tl.load(scale_ptr + chan, mask=mask)
    softmax_val = tl.exp(x_vals - global_max) / global_sum * scale_val
    tl.store(out_ptr + offsets, softmax_val, mask=mask)

def triton_softmax_clamp_scale(x: torch.Tensor, scale: torch.Tensor, clamp_min, clamp_max):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    b, c, d, h, w = x.shape
    N = d * h * w
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    temp_max = torch.empty(b, c, num_blocks, dtype=x.dtype, device=x.device)
    temp_sum = torch.empty(b, c, num_blocks, dtype=x.dtype, device=x.device)
    global_max = torch.empty(b * c, dtype=x.dtype, device=x.device)
    global_sum = torch.empty(b * c, dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    scale_flat = scale.view(-1)
    grid1 = b * c * num_blocks
    compute_local_max_kernel[grid1](x, temp_max, b, c, d, h, w, BLOCK_SIZE=BLOCK_SIZE, num_blocks=num_blocks)
    grid2 = b * c
    reduce_max_kernel[grid2](temp_max, global_max, num_blocks=num_blocks, BLOCK_SIZE=num_blocks)
    compute_local_sum_kernel[grid1](x, temp_sum, global_max, b, c, d, h, w, BLOCK_SIZE=BLOCK_SIZE, num_blocks=num_blocks)
    reduce_sum_kernel[grid2](temp_sum, global_sum, num_blocks=num_blocks, BLOCK_SIZE=num_blocks)
    grid3 = ((b * c * d * h * w + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    compute_softmax_kernel[grid3](x, out, scale_flat, global_max, global_sum, clamp_min, clamp_max, b, c, d, h, w, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Model that performs average pooling, 3D transposed convolution, clamping,
    spatial softmax, and multiplication by a learnable scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = triton_softmax_clamp_scale(x, self.scale, self.clamp_min, self.clamp_max)
        return x