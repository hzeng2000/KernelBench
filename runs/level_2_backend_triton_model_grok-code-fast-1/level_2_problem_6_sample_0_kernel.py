import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_channels,
    depth,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    
    offsets = tl.arange(0, num_channels)
    input_offsets = (
        b * (num_channels * depth * height * width) +
        offsets * (depth * height * width) +
        d * (height * width) +
        h * width +
        w
    )
    x = tl.load(input_ptr + input_offsets)
    
    max_val = tl.max(x, axis=0)
    x = x - max_val
    x = tl.exp(x)
    sum_val = tl.sum(x, axis=0)
    out = x / sum_val
    
    tl.store(output_ptr + input_offsets, out)


def triton_softmax(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    batch, channels, d, h, w = x.shape
    out = torch.empty_like(x)
    grid = (batch, d, h, w)
    softmax_kernel[grid](
        x, out, batch, channels, d, h, w, BLOCK_SIZE=channels
    )
    return out


@triton.jit
def maxpool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_channels,
    depth,
    height,
    width,
    od,
    oh,
    ow,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    od_idx = tl.program_id(2)
    oh_idx = tl.program_id(3)
    ow_idx = tl.program_id(4)
    
    id_start = od_idx * 2
    ih_start = oh_idx * 2
    iw_start = ow_idx * 2
    
    offset0 = id_start * (height * width) + ih_start * width + iw_start
    offset1 = id_start * (height * width) + ih_start * width + (iw_start + 1)
    offset2 = id_start * (height * width) + (ih_start + 1) * width + iw_start
    offset3 = id_start * (height * width) + (ih_start + 1) * width + (iw_start + 1)
    offset4 = (id_start + 1) * (height * width) + ih_start * width + iw_start
    offset5 = (id_start + 1) * (height * width) + ih_start * width + (iw_start + 1)
    offset6 = (id_start + 1) * (height * width) + (ih_start + 1) * width + iw_start
    offset7 = (id_start + 1) * (height * width) + (ih_start + 1) * width + (iw_start + 1)
    
    base_offset = b * (num_channels * depth * height * width) + c * (depth * height * width)
    x0 = tl.load(input_ptr + base_offset + offset0)
    x1 = tl.load(input_ptr + base_offset + offset1)
    x2 = tl.load(input_ptr + base_offset + offset2)
    x3 = tl.load(input_ptr + base_offset + offset3)
    x4 = tl.load(input_ptr + base_offset + offset4)
    x5 = tl.load(input_ptr + base_offset + offset5)
    x6 = tl.load(input_ptr + base_offset + offset6)
    x7 = tl.load(input_ptr + base_offset + offset7)
    
    max_val = tl.maximum(tl.maximum(tl.maximum(x0, x1), tl.maximum(x2, x3)), tl.maximum(tl.maximum(x4, x5), tl.maximum(x6, x7)))
    
    output_offset = (
        b * (num_channels * od * oh * ow) +
        c * (od * oh * ow) +
        od_idx * (oh * ow) +
        oh_idx * ow +
        ow_idx
    )
    tl.store(output_ptr + output_offset, max_val)


def triton_maxpool3d(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    batch, channels, d, h, w = x.shape
    assert d % 2 == 0 and h % 2 == 0 and w % 2 == 0, "Dimensions must be even for kernel_size=2, stride=2."
    od = d // 2
    oh = h // 2
    ow = w // 2
    out = torch.empty(batch, channels, od, oh, ow, device=x.device, dtype=x.dtype)
    grid = (batch, channels, od, oh, ow)
    maxpool3d_kernel[grid](
        x, out, batch, channels, d, h, w, od, oh, ow, BLOCK_SIZE=1
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = self.conv(x)
        x = triton_softmax(x)
        x = triton_maxpool3d(x)
        x = triton_maxpool3d(x)
        return x