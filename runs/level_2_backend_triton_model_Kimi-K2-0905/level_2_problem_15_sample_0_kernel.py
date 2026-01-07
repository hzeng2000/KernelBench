import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_bn_mean_sub_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    batch_size, channels, depth, height, width,
    eps, momentum,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = batch_size * channels * depth * height * width
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute indices from linear offset
    w = offsets % width
    tmp = offsets // width
    h = tmp % height
    tmp = tmp // height
    d = tmp % depth
    tmp = tmp // depth
    c = tmp % channels
    b = tmp // channels

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # BatchNorm parameters
    weight = tl.load(weight_ptr + c, mask=c < channels, other=1.0)
    bias = tl.load(bias_ptr + c, mask=c < channels, other=0.0)
    running_mean = tl.load(running_mean_ptr + c, mask=c < channels, other=0.0)
    running_var = tl.load(running_var_ptr + c, mask=c < channels, other=1.0)

    # BatchNorm inference: (x - running_mean) / sqrt(running_var + eps) * weight + bias
    var_eps = running_var + eps
    inv_std = tl.rsqrt(var_eps)
    x_norm = (x - running_mean) * inv_std
    x_bn = x_norm * weight + bias

    # Subtract spatial mean
    # Compute spatial mean per sample and channel
    spatial_size = depth * height * width
    spatial_offset = b * channels * spatial_size + c * spatial_size
    spatial_base = spatial_offset
    spatial_end = spatial_base + spatial_size

    # Sum reduction over spatial dims
    sum_val = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
    for i in range(0, spatial_size, BLOCK_SIZE):
        spatial_offs = spatial_base + i + tl.arange(0, BLOCK_SIZE)
        spatial_mask = spatial_offs < spatial_end
        val = tl.load(x_ptr + spatial_offs, mask=spatial_mask, other=0.0)
        sum_val += val
    total = tl.sum(sum_val)
    mean = total / spatial_size

    out = x_bn - mean
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.eps = self.batch_norm.eps

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse BN + mean subtraction
        out = torch.empty_like(x)
        B, C, D, H, W = x.shape
        n_elements = x.numel()
        BLOCK_SIZE = 256
        grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        fused_transpose_bn_mean_sub_kernel[grid](
            x, out,
            self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var,
            B, C, D, H, W,
            self.eps, self.batch_norm.momentum,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out