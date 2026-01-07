import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_bias_scale_sigmoid_kernel(
    x_ptr, w_ptr, b_ptr, s_ptr, out_ptr,
    batch, in_c, out_c, h, w, k,
    stride_h, stride_w, pad_h, pad_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw = pid_hw * BLOCK_H * BLOCK_W + tl.arange(0, BLOCK_H * BLOCK_W)
    oh = hw // ((w + 2 * pad_w - k) // stride_w + 1)
    ow = hw % ((w + 2 * pad_w - k) // stride_w + 1)

    if oh >= (h + 2 * pad_h - k) // stride_h + 1:
        return

    acc = tl.zeros([BLOCK_H * BLOCK_W], dtype=tl.float32)
    for ic in range(in_c):
        for kh in range(k):
            for kw in range(k):
                ih = oh * stride_h - pad_h + kh
                iw = ow * stride_w - pad_w + kw
                mask = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
                x_idx = pid_b * in_c * h * w + ic * h * w + ih * w + iw
                x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                w_idx = pid_oc * in_c * k * k + ic * k * k + kh * k + kw
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    b_val = tl.load(b_ptr + pid_oc)
    s_val = tl.load(s_ptr + pid_oc)
    out_val = acc + b_val
    out_val = out_val * s_val
    out_val = tl.sigmoid(out_val)

    out_h = (h + 2 * pad_h - k) // stride_h + 1
    out_w = (w + 2 * pad_w - k) // stride_w + 1
    out_idx = pid_b * out_c * out_h * out_w + pid_oc * out_h * out_w + oh * out_w + ow
    tl.store(out_ptr + out_idx, out_val)


@triton.jit
def group_norm_kernel(
    x_ptr, out_ptr, gamma_ptr, beta_ptr,
    batch, groups, channels, h, w,
    eps, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    group_size = channels // groups * h * w
    offset = pid * group_size + tl.arange(0, BLOCK_SIZE)

    mask = offset < (pid + 1) * group_size
    x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)

    mean = tl.sum(x_val, axis=0) / group_size
    var = tl.sum((x_val - mean) * (x_val - mean), axis=0) / group_size
    inv_std = tl.rsqrt(var + eps)

    c_per_g = channels // groups
    g = (offset // (h * w)) // c_per_g
    c_in_g = (offset // (h * w)) % c_per_g

    gamma = tl.load(gamma_ptr + g * c_per_g + c_in_g)
    beta = tl.load(beta_ptr + g * c_per_g + c_in_g)

    out_val = (x_val - mean) * inv_std * gamma + beta
    tl.store(out_ptr + offset, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups

    def forward(self, x):
        batch, in_c, h, w = x.shape
        out_c = self.conv.out_channels
        k = self.conv.kernel_size[0]
        stride_h = stride_w = self.conv.stride[0]
        pad_h = pad_w = self.conv.padding[0]

        out_h = (h + 2 * pad_h - k) // stride_h + 1
        out_w = (w + 2 * pad_w - k) // stride_w + 1

        x_conv = torch.empty(batch, out_c, out_h, out_w, device=x.device, dtype=x.dtype)
        BLOCK_H = 8
        BLOCK_W = 8
        grid = (batch, out_c, (out_h * out_w + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W))
        fused_conv_bias_scale_sigmoid_kernel[grid](
            x, self.conv.weight, self.bias.squeeze(), self.scale.squeeze(), x_conv,
            batch, in_c, out_c, h, w, k,
            stride_h, stride_w, pad_h, pad_w,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )

        # GroupNorm
        x_norm = torch.empty_like(x_conv)
        channels = out_c
        groups = self.num_groups
        eps = 1e-5
        BLOCK_SIZE = 256
        num_blocks = batch * groups
        gamma = self.group_norm.weight.view(groups, -1).repeat(batch, 1).view(-1)
        beta = self.group_norm.bias.view(groups, -1).repeat(batch, 1).view(-1)
        group_norm_kernel[(num_blocks + BLOCK_SIZE - 1) // BLOCK_SIZE](
            x_conv.view(-1), x_norm.view(-1), gamma, beta,
            batch, groups, channels, out_h, out_w,
            eps, BLOCK_SIZE=BLOCK_SIZE
        )

        return x_norm