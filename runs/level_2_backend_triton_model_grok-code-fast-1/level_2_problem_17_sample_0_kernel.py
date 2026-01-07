import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reduce_kernel(
    x_ptr,
    sum_ptr,
    sum_sq_ptr,
    n, c, h, w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)
    offset = pid_n * c * h * w + pid_c * h * w
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < h * w
    x_offsets = offset + offsets
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    sum_val = tl.sum(x)
    sum_sq_val = tl.sum(x * x)
    tl.atomic_add(sum_ptr + pid_n * c + pid_c, sum_val)
    tl.atomic_add(sum_sq_ptr + pid_n * c + pid_c, sum_sq_val)

@triton.jit
def normalize_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    divide_by,
    n, c, h, w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_block = tl.program_id(2)
    offset = pid_n * c * h * w + pid_c * h * w
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < h * w
    x_offsets = offset + offsets
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + pid_n * c + pid_c)
    var = tl.load(var_ptr + pid_n * c + pid_c)
    out = (x - mean) / (tl.sqrt(var + 1e-5) * divide_by)
    tl.store(out_ptr + x_offsets, out, mask=mask)

def triton_instance_norm_divide(x: torch.Tensor, divide_by: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    n, c, h, w = x.shape
    sum_buffer = torch.zeros(n, c, device=x.device, dtype=x.dtype)
    sum_sq_buffer = torch.zeros(n, c, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid_reduce = (n, c, (h * w + BLOCK_SIZE - 1) // BLOCK_SIZE)
    reduce_kernel[grid_reduce](x, sum_buffer, sum_sq_buffer, n, c, h, w, BLOCK_SIZE=BLOCK_SIZE)
    mean = sum_buffer / (h * w)
    var = sum_sq_buffer / (h * w) - mean * mean
    out = torch.empty_like(x)
    grid_norm = (n, c, (h * w + BLOCK_SIZE - 1) // BLOCK_SIZE)
    normalize_kernel[grid_norm](x, out, mean, var, divide_by, n, c, h, w, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = triton_instance_norm_divide(x, self.divide_by)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]