import torch
import torch.nn as nn
import triton
import triton.language as tl

TILE_H = 32
TILE_W = 32

@triton.jit
def gelu_sum_tiles_kernel(
    x_ptr, temp_ptr, B, C, H, W, TILE_H: tl.constexpr, TILE_W: tl.constexpr, num_tiles_h: tl.constexpr, num_tiles_w: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    tile_id = tl.program_id(2)
    tile_h = tile_id // num_tiles_w
    tile_w = tile_id % num_tiles_w
    h_start = tile_h * TILE_H
    w_start = tile_w * TILE_W
    h_offsets = h_start + tl.arange(0, TILE_H)
    w_offsets = w_start + tl.arange(0, TILE_W)
    offsets = b * (C * H * W) + c * (H * W) + h_offsets[:, None] * W + w_offsets[None, :]
    x_tile = tl.load(x_ptr + offsets)
    gelu_tile = 0.5 * x_tile * (1 + tl.erf(x_tile / tl.sqrt(2.0)))
    sum_tile = tl.sum(gelu_tile)
    temp_offset = b * (C * (num_tiles_h * num_tiles_w)) + c * (num_tiles_h * num_tiles_w) + tile_id
    tl.store(temp_ptr + temp_offset, sum_tile)

@triton.jit
def reduce_tiles_kernel(
    temp_ptr, out_ptr, B, C, num_tiles, HW
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    offsets = b * (C * num_tiles) + c * num_tiles + tl.arange(0, num_tiles)
    temp_vals = tl.load(temp_ptr + offsets)
    total_sum = tl.sum(temp_vals)
    avg = total_sum / HW
    out_offset = b * C + c
    tl.store(out_ptr + out_offset, avg)

def triton_fused_gelu_avg(x: torch.Tensor):
    B, C, H, W = x.shape
    assert H % TILE_H == 0 and W % TILE_W == 0
    num_tiles_h = H // TILE_H
    num_tiles_w = W // TILE_W
    num_tiles = num_tiles_h * num_tiles_w
    temp = torch.empty(B, C, num_tiles, dtype=x.dtype, device=x.device)
    grid1 = (B, C, num_tiles)
    gelu_sum_tiles_kernel[grid1](x, temp, B, C, H, W, TILE_H, TILE_W, num_tiles_h, num_tiles_w)
    out = torch.empty(B, C, dtype=x.dtype, device=x.device)
    grid2 = (B, C)
    HW = H * W
    reduce_tiles_kernel[grid2](temp, out, B, C, num_tiles, HW)
    return out

class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.conv(x)
        x = triton_fused_gelu_avg(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]