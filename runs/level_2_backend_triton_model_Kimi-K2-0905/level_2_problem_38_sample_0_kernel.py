import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels, in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)

    # Compute output spatial indices
    od = pid_d
    oh = tl.program_id(3)
    ow = tl.program_id(4)

    # Compute input start indices
    id_start = od * stride_d
    ih_start = oh * stride_h
    iw_start = ow * stride_w

    # Accumulate over kernel
    acc = 0.0
    count = 0
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                id = id_start + kd
                ih = ih_start + kh
                iw = iw_start + kw
                if id < in_d and ih < in_h and iw < in_w:
                    idx = ((pid_b * in_channels + pid_c) * in_d + id) * in_h * in_w + ih * in_w + iw
                    val = tl.load(x_ptr + idx)
                    acc += val
                    count += 1
    out_idx = ((pid_b * in_channels + pid_c) * out_d + od) * out_h * out_w + oh * out_w + ow
    tl.store(out_ptr + out_idx, acc / count)


@triton.jit
def clamp_kernel(x_ptr, out_ptr, n_elements, min_val, max_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.clamp(x, min_val, max_val)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def spatial_softmax_kernel(x_ptr, out_ptr, b_stride, c_stride, spatial_size, BLOCK_SPATIAL: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    base_offset = pid_b * b_stride + pid_c * c_stride
    offsets = base_offset + tl.arange(0, BLOCK_SPATIAL)
    mask = tl.arange(0, BLOCK_SPATIAL) < spatial_size

    # Load and compute max for numerical stability
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_stable = x - x_max

    # Compute softmax
    exp_x = tl.exp(x_stable)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_avg_pool3d(x, kernel_size, stride):
    b, c, d, h, w = x.shape
    out_d = (d - kernel_size) // stride + 1
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    out = torch.empty(b, c, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

    grid = (b, c, out_d, out_h, out_w)
    avg_pool3d_kernel[grid](
        x, out,
        b, c, d, h, w,
        out_d, out_h, out_w,
        kernel_size, kernel_size, kernel_size,
        stride, stride, stride,
        BLOCK_D=1, BLOCK_H=1, BLOCK_W=1
    )
    return out


def triton_clamp(x, min_val, max_val):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    clamp_kernel[grid](x, out, n_elements, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_spatial_softmax(x):
    b, c, d, h, w = x.shape
    spatial_size = d * h * w
    x_flat = x.view(b, c, -1)
    out_flat = torch.empty_like(x_flat)

    BLOCK_SPATIAL = triton.next_power_of_2(spatial_size)
    grid = (b, c)
    spatial_softmax_kernel[grid](
        x_flat, out_flat,
        x_flat.stride(0), x_flat.stride(1), spatial_size,
        BLOCK_SPATIAL=BLOCK_SPATIAL
    )
    return out_flat.view(b, c, d, h, w)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = triton_avg_pool3d(x, self.pool_kernel_size, self.pool_kernel_size)
        x = self.conv_transpose(x)
        x = triton_clamp(x, self.clamp_min, self.clamp_max)
        x = triton_spatial_softmax(x)
        x = x * self.scale
        return x