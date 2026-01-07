import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_spatial_kernel(
    x_ptr,
    sum_ptr,  # shape (batch_size, out_channels)
    B, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    block_id = tl.program_id(2)
    spatial_size = D * H * W
    spatial_start = block_id * BLOCK_SIZE
    offsets_spatial = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask_spatial = offsets_spatial < spatial_size
    stride_b = C * D * H * W
    stride_c = D * H * W
    offsets = b * stride_b + c * stride_c + offsets_spatial
    x_vals = tl.load(x_ptr + offsets, mask=mask_spatial, other=0.0)
    block_sum = tl.sum(x_vals, axis=0)
    tl.atomic_add(sum_ptr + b * C + c, block_sum)


@triton.jit
def subtract_mean_kernel(
    x_ptr,
    mean_ptr,  # (B, C)
    out_ptr,
    B, C, D, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total_elements = B * C * D * H * W
    mask = offsets < total_elements
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    stride_b = C * D * H * W
    stride_c = D * H * W
    b = offsets // stride_b
    c = (offsets % stride_b) // stride_c
    mean_vals = tl.load(mean_ptr + b * C + c, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x_vals - mean_vals, mask=mask)


def triton_subtract_mean(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    B, C, D, H, W = x.shape
    spatial_size = D * H * W
    sum_tensor = torch.zeros(B, C, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 128
    num_blocks_spatial = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_sum = (B, C, num_blocks_spatial)
    sum_spatial_kernel[grid_sum](x, sum_tensor, B, C, D, H, W, BLOCK_SIZE=BLOCK_SIZE)
    mean_tensor = sum_tensor / spatial_size
    out = torch.empty_like(x)
    total_elements = x.numel()
    BLOCK_SIZE_out = 128
    grid_out = ((total_elements + BLOCK_SIZE_out - 1) // BLOCK_SIZE_out,)
    subtract_mean_kernel[grid_out](x, mean_tensor, out, B, C, D, H, W, BLOCK_SIZE=BLOCK_SIZE_out)
    return out


class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = triton_subtract_mean(x)
        return x