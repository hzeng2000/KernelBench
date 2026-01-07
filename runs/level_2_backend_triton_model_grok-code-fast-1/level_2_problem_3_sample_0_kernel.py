import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def add_scalar_kernel(
    x_ptr,
    scalar,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x + scalar
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add_scalar(x: torch.Tensor, scalar: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_scalar_kernel[grid](x, scalar, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 0.5 * x * (1.0 + tl.libdevice.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def avg_pool3d_kernel(
    x_ptr,
    out_ptr,
    B, C, D, H, W,
    x_stride_b, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    out_stride_b, out_stride_c, out_stride_od, out_stride_oh, out_stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_od = D // 2
    num_oh = H // 2
    num_ow = W // 2
    total_per_b = C * num_od * num_oh * num_ow
    b = pid // total_per_b
    rem = pid % total_per_b
    total_per_c = num_od * num_oh * num_ow
    c = rem // total_per_c
    rem = rem % total_per_c
    total_per_od = num_oh * num_ow
    od = rem // total_per_od
    rem = rem % total_per_od
    oh = rem // num_ow
    ow = rem % num_ow
    id_start = od * 2
    ih_start = oh * 2
    iw_start = ow * 2
    sum_val = 0.0
    for dd in range(2):
        for dh in range(2):
            for dw in range(2):
                id = id_start + dd
                ih = ih_start + dh
                iw = iw_start + dw
                offset = (b * x_stride_b + c * x_stride_c + id * x_stride_d + ih * x_stride_h + iw * x_stride_w)
                val = tl.load(x_ptr + offset)
                sum_val += val
    out_val = sum_val / 8.0
    out_offset = b * out_stride_b + c * out_stride_c + od * out_stride_od + oh * out_stride_oh + ow * out_stride_ow
    tl.store(out_ptr + out_offset, out_val)


def triton_avg_pool3d(x: torch.Tensor, kernel_size):
    assert x.is_cuda and len(kernel_size) == 3 and kernel_size == (2, 2, 2), "Input must be CUDA tensor with kernel_size (2,2,2)."
    x = x.contiguous()
    B, C, D, H, W = x.shape
    out = torch.empty(B, C, D // 2, H // 2, W // 2, dtype=x.dtype, device=x.device)
    x_stride = x.stride()
    out_stride = out.stride()
    total_elements = B * C * (D // 2) * (H // 2) * (W // 2)
    BLOCK_SIZE = 1  # Since each kernel handles one output element
    grid = (total_elements,)
    avg_pool3d_kernel[grid](
        x, out, B, C, D, H, W,
        x_stride[0], x_stride[1], x_stride[2], x_stride[3], x_stride[4],
        out_stride[0], out_stride[1], out_stride[2], out_stride[3], out_stride[4],
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_add_scalar(x, self.sum_weight.item())
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = triton_avg_pool3d(x, self.pool_kernel_size)
        x = triton_gelu(x)
        return x