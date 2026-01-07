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
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale(x: torch.Tensor, scale: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    scale_kernel[grid](x, out, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def global_avg_clamp_kernel(
    x_ptr,
    out_ptr,
    batch_stride,
    c_stride,
    d_stride,
    h_stride,
    w_stride,
    num_batch,
    num_c,
    num_d,
    num_h,
    num_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // num_c
    c = pid % num_c
    if batch >= num_batch or c >= num_c:
        return
    sum_val = 0.0
    num_elements = num_d * num_h * num_w
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        d = offsets // (num_h * num_w)
        rem = offsets % (num_h * num_w)
        h = rem // num_w
        w = rem % num_w
        idx = (batch * batch_stride + c * c_stride + d * d_stride + h * h_stride + w * w_stride)
        val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(val)
    avg = sum_val / num_elements
    clamped = tl.maximum(0.0, tl.minimum(1.0, avg))
    out_idx = batch * num_c + c
    tl.store(out_ptr + out_idx, clamped)


def triton_global_avg_clamp(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    batch, c, d, h, w = x.shape
    out = torch.empty(batch, c, 1, 1, 1, dtype=x.dtype, device=x.device)
    batch_stride = c * d * h * w
    c_stride = d * h * w
    d_stride = h * w
    h_stride = w
    w_stride = 1
    BLOCK_SIZE = 1024
    grid = (batch * c,)
    global_avg_clamp_kernel[grid](
        x, out, batch_stride, c_stride, d_stride, h_stride, w_stride,
        batch, c, d, h, w, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_scale(x, self.scale)
        x = self.maxpool(x)
        x = triton_global_avg_clamp(x)
        return x