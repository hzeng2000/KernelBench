import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    x_ptr,
    out_ptr,
    B, C, D, H, W,
    out_D, out_H, out_W,
    kernel_size,
    stride,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    od = tl.program_id(2)
    oh = tl.program_id(3)
    ow = tl.program_id(4)

    id_start = od * stride
    ih_start = oh * stride
    iw_start = ow * stride

    d_offsets = id_start + tl.arange(0, kernel_size)
    h_offsets = ih_start + tl.arange(0, kernel_size)
    w_offsets = iw_start + tl.arange(0, kernel_size)

    d_mask = d_offsets < D
    h_mask = h_offsets < H
    w_mask = w_offsets < W

    base = b * C * D * H * W + c * D * H * W

    total = 0.0
    count = 0.0
    for dd in range(kernel_size):
        for hh in range(kernel_size):
            for ww in range(kernel_size):
                mask = d_mask[dd] & h_mask[hh] & w_mask[ww]
                offset = d_offsets[dd] * H * W + h_offsets[hh] * W + w_offsets[ww]
                val = tl.load(x_ptr + base + offset, mask=mask, other=0.0)
                total += val
                count += tl.where(mask, 1.0, 0.0)

    avg = total / count

    out_base = b * C * out_D * out_H * out_W + c * out_D * out_H * out_W + od * out_H * out_W + oh * out_W + ow
    tl.store(out_ptr + out_base, avg)


def triton_avg_pool3d(x, kernel_size=4, stride=4):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    B, C, D, H, W = x.shape
    out_D = (D - kernel_size) // stride + 1
    out_H = (H - kernel_size) // stride + 1
    out_W = (W - kernel_size) // stride + 1
    out = torch.empty(B, C, out_D, out_H, out_W, device=x.device, dtype=x.dtype)
    grid = (B, C, out_D, out_H, out_W)
    avg_pool_kernel[grid](x, out, B, C, D, H, W, out_D, out_H, out_W, kernel_size, stride)
    return out


class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    and a fused average pooling (equivalent to two average pooling layers with kernel_size=2).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = triton_avg_pool3d(x, kernel_size=4, stride=4)
        return x