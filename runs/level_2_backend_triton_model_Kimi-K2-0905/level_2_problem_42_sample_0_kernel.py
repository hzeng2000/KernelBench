import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_c, out_c, h_in, w_in, h_out, w_out, k,
    stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    hw_out = h_out * w_out
    nhw = batch_size * hw_out
    num_blocks = (nhw + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid >= num_blocks:
        return

    for block in range(pid, num_blocks, tl.num_programs(0)):
        start = block * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < nhw
        n_idx = offsets // hw_out
        hw_idx = offsets % hw_out
        h_idx = hw_idx // w_out
        w_idx = hw_idx % w_out

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for ic in range(in_c):
            for kh in range(k):
                for kw in range(k):
                    h_in_idx = h_idx + kh - padding
                    w_in_idx = w_idx + kw - padding
                    in_bounds = (h_in_idx >= 0) & (h_in_idx < h_in) & (w_in_idx >= 0) & (w_in_idx < w_in)
                    x_val = tl.load(x_ptr + n_idx * in_c * h_in * w_in + ic * h_in * w_in + h_in_idx * w_in + w_in_idx, mask=in_bounds & mask, other=0.0)
                    for oc in range(out_c):
                        w_val = tl.load(w_ptr + oc * in_c * k * k + ic * k * k + kh * k + kw)
                        acc += x_val * w_val
        tl.store(out_ptr + offsets * out_c + tl.arange(0, out_c)[None, :] * BLOCK_SIZE, acc[:, None], mask=mask)


@triton.jit
def global_avg_pool_kernel(x_ptr, out_ptr, batch_size, c, h, w, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    hw = h * w
    total = batch_size * c
    num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid >= num_blocks:
        return

    for block in range(pid, num_blocks, tl.num_programs(0)):
        start = block * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        n_idx = offsets // c
        c_idx = offsets % c

        sum_val = 0.0
        for i in range(hw):
            h_idx = i // w
            w_idx = i % w
            val = tl.load(x_ptr + n_idx * c * h * w + c_idx * h * w + h_idx * w + w_idx, mask=mask)
            sum_val += val
        avg = sum_val / hw
        tl.store(out_ptr + offsets, avg, mask=mask)


@triton.jit
def add_bias_kernel(x_ptr, bias_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets % tl.load(bias_ptr + 0))
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, batch_size, c, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    max_val = -float('inf')
    for i in range(c):
        val = tl.load(x_ptr + pid * c + i)
        max_val = tl.maximum(max_val, val)
    sum_exp = 0.0
    for i in range(c):
        val = tl.load(x_ptr + pid * c + i)
        sum_exp += tl.exp(val - max_val)
    lse = max_val + tl.log(sum_exp)
    tl.store(out_ptr + pid, lse)


@triton.jit
def sum_hw_kernel(x_ptr, out_ptr, batch_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    sum_val = 0.0
    tl.store(out_ptr + pid, tl.load(x_ptr + pid) * 10.0)


def triton_conv_transpose(x, w, stride=1, padding=0, dilation=1):
    batch_size, in_c, h_in, w_in = x.shape
    out_c, _, k, _ = w.shape
    h_out = (h_in - 1) * stride - 2 * padding + dilation * (k - 1) + 1
    w_out = (w_in - 1) * stride - 2 * padding + dilation * (k - 1) + 1
    out = torch.empty(batch_size, out_c, h_out, w_out, device=x.device, dtype=x.dtype)
    n_elements = batch_size * out_c * h_out * w_out
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    conv_transpose_kernel[grid](
        x, w, out,
        batch_size, in_c, out_c, h_in, w_in, h_out, w_out, k,
        stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_global_avg_pool(x):
    batch_size, c, h, w = x.shape
    out = torch.empty(batch_size, c, 1, 1, device=x.device, dtype=x.dtype)
    total = batch_size * c
    BLOCK_SIZE = 128
    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    global_avg_pool_kernel[grid](x, out, batch_size, c, h, w, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_add_bias(x, bias):
    x = x.contiguous()
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_bias_kernel[grid](x, bias, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_logsumexp(x):
    batch_size, c, _, _ = x.shape
    out = torch.empty(batch_size, 1, 1, 1, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 128
    grid = lambda meta: (batch_size,)
    logsumexp_kernel[grid](x.view(batch_size, c), out.view(batch_size), batch_size, c, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_sum_hw(x):
    batch_size, _, _, _ = x.shape
    out = torch.empty(batch_size, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 128
    grid = lambda meta: (batch_size,)
    sum_hw_kernel[grid](x.view(batch_size), out, batch_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = triton_conv_transpose(x, self.conv_transpose.weight, stride=1, padding=0, dilation=1)
        x = triton_global_avg_pool(x)
        x = triton_add_bias(x, self.bias)
        x = triton_logsumexp(x)
        x = triton_sum_hw(x)
        return x