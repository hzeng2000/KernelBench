import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def group_norm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr, N, C, DHW, groups, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n = pid // groups
    g = pid % groups
    c_per_g = C // groups
    start_c = g * c_per_g
    end_c = start_c + c_per_g
    dhw = DHW

    # Compute mean
    sum_val = 0.0
    for c in range(start_c, end_c):
        for i in range(0, dhw, BLOCK_SIZE):
            offsets = (n * C + c) * dhw + i + tl.arange(0, BLOCK_SIZE)
            mask = i + tl.arange(0, BLOCK_SIZE) < dhw
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            sum_val += tl.sum(x)
    mean = sum_val / (c_per_g * dhw)

    # Compute var
    var_sum = 0.0
    for c in range(start_c, end_c):
        for i in range(0, dhw, BLOCK_SIZE):
            offsets = (n * C + c) * dhw + i + tl.arange(0, BLOCK_SIZE)
            mask = i + tl.arange(0, BLOCK_SIZE) < dhw
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            var_sum += tl.sum((x - mean) * (x - mean))
    var = var_sum / (c_per_g * dhw)
    rstd = tl.rsqrt(var + eps)

    # Apply norm, weight, bias
    for c in range(start_c, end_c):
        w = tl.load(weight_ptr + c)
        b = tl.load(bias_ptr + c)
        for i in range(0, dhw, BLOCK_SIZE):
            offsets = (n * C + c) * dhw + i + tl.arange(0, BLOCK_SIZE)
            mask = i + tl.arange(0, BLOCK_SIZE) < dhw
            x = tl.load(x_ptr + offsets, mask=mask)
            out = (x - mean) * rstd * w + b
            tl.store(out_ptr + offsets, out, mask=mask)


def triton_group_norm(x, weight, bias, groups, eps):
    assert x.is_cuda
    x = x.contiguous()
    N, C, D, H, W = x.shape
    DHW = D * H * W
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = (N * groups,)
    group_norm_kernel[grid](
        x, out, weight, bias, N, C, DHW, groups, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


@triton.jit
def hardswish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * tl.clamp(x + 3.0, 0.0, 6.0) / 6.0
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardswish(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_swish(x)
        # Extract weight and bias from group norm
        weight = self.group_norm.weight
        bias = self.group_norm.bias
        x = triton_group_norm(x, weight, bias, self.groups, self.eps)
        x = triton_hardswish(x)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]