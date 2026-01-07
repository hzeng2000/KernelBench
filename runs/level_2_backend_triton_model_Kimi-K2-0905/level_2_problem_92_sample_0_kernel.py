import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, h, w, k,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    hw = pid_hw
    oh = hw // ((w + 2 * pad_w - k) // stride_w + 1)
    ow = hw % ((w + 2 * pad_w - k) // stride_w + 1)

    acc = 0.0
    for ic in range(in_c):
        for kh in range(k):
            for kw in range(k):
                ih = oh * stride_h - pad_h + kh
                iw = ow * stride_w - pad_w + kw
                if ih >= 0 and ih < h and iw >= 0 and iw < w:
                    x_val = tl.load(x_ptr + pid_b * in_c * h * w + ic * h * w + ih * w + iw)
                    w_val = tl.load(w_ptr + pid_c * in_c * k * k + ic * k * k + kh * k + kw)
                    acc += x_val * w_val
    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_c)
        acc += b_val
    out_h = (h + 2 * pad_h - k) // stride_h + 1
    out_w = (w + 2 * pad_w - k) // stride_w + 1
    tl.store(out_ptr + pid_b * out_c * out_h * out_w + pid_c * out_h * out_w + oh * out_w + ow, acc)


@triton.jit
def group_norm_tanh_hardswish_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    batch, c, h, w, groups,
    eps,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_hw = tl.program_id(2)

    start_c = pid_g * (c // groups)
    end_c = (pid_g + 1) * (c // groups)
    hw = pid_hw
    chw = c * h * w

    # Compute mean
    sum_val = 0.0
    count = 0
    for ci in range(start_c, end_c):
        idx = pid_b * chw + ci * h * w + hw
        val = tl.load(x_ptr + idx)
        sum_val += val
        count += 1
    mean = sum_val / count

    # Compute var
    var_sum = 0.0
    for ci in range(start_c, end_c):
        idx = pid_b * chw + ci * h * w + hw
        val = tl.load(x_ptr + idx)
        var_sum += (val - mean) * (val - mean)
    var = var_sum / count
    rstd = tl.rsqrt(var + eps)

    # Normalize, apply tanh and hardswish
    for ci in range(start_c, end_c):
        idx = pid_b * chw + ci * h * w + hw
        val = tl.load(x_ptr + idx)
        norm_val = (val - mean) * rstd
        if weight_ptr is not None:
            norm_val *= tl.load(weight_ptr + ci)
        if bias_ptr is not None:
            norm_val += tl.load(bias_ptr + ci)
        tanh_val = tl.tanh(norm_val)
        hard_swish_val = tanh_val * tl.min(tl.max(norm_val + 3.0, 0.0), 6.0) / 6.0
        tl.store(out_ptr + idx, hard_swish_val)


@triton.jit
def residual_logsumexp_kernel(
    a_ptr, b_ptr, out_ptr,
    batch, c, h, w,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw = pid_hw
    chw = c * h * w
    max_val = -float('inf')
    # First pass: find max
    for ci in range(c):
        idx = pid_b * chw + ci * h * w + hw
        a = tl.load(a_ptr + idx)
        b = tl.load(b_ptr + idx)
        val = a + b
        max_val = tl.max(max_val, val)
    # Second pass: compute sum exp
    sum_exp = 0.0
    for ci in range(c):
        idx = pid_b * chw + ci * h * w + hw
        a = tl.load(a_ptr + idx)
        b = tl.load(b_ptr + idx)
        val = a + b
        sum_exp += tl.exp(val - max_val)
    logsumexp = max_val + tl.log(sum_exp)
    tl.store(out_ptr + pid_b * h * w + hw, logsumexp)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0):
    batch, in_c, h, w = x.shape
    out_c, _, k, _ = weight.shape
    out_h = (h + 2 * padding - k) // stride + 1
    out_w = (w + 2 * padding - k) // stride + 1
    out = torch.empty(batch, out_c, out_h, out_w, device=x.device, dtype=x.dtype)

    grid = (batch, out_c, out_h * out_w)
    BLOCK_H = 4
    BLOCK_W = 4
    conv_kernel[grid](
        x, weight, bias, out,
        batch, in_c, out_c, h, w, k,
        stride, stride, padding, padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return out


def triton_group_norm_tanh_hardswish(x, weight, bias, groups, eps):
    batch, c, h, w = x.shape
    out = torch.empty_like(x)

    grid = (batch, groups, h * w)
    BLOCK_C = c // groups
    BLOCK_HW = h * w
    group_norm_tanh_hardswish_kernel[grid](
        x, out, weight, bias,
        batch, c, h, w, groups,
        eps,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW
    )
    return out


def triton_residual_logsumexp(a, b):
    batch, c, h, w = a.shape
    out = torch.empty(batch, 1, h, w, device=a.device, dtype=a.dtype)

    grid = (batch, h * w)
    BLOCK_C = c
    BLOCK_HW = h * w
    residual_logsumexp_kernel[grid](
        a, b, out,
        batch, c, h, w,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.group_norm_weight = nn.Parameter(torch.randn(out_channels))
        self.group_norm_bias = nn.Parameter(torch.randn(out_channels))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        x_conv = triton_conv2d(x, self.conv_weight, self.conv_bias, stride=1, padding=0)
        x_norm = triton_group_norm_tanh_hardswish(x_conv, self.group_norm_weight, self.group_norm_bias, self.groups, self.eps)
        x_res = triton_residual_logsumexp(x_conv, x_norm)
        return x_res


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]