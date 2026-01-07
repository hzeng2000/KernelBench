import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_gn_scale_max_clamp_kernel(
    x_ptr, weight_ptr, bias_ptr, gn_weight_ptr, gn_bias_ptr, scale_ptr,
    out_ptr, running_mean_ptr, running_var_ptr,
    batch_size, in_channels, out_channels, height, width, kernel_size,
    num_groups, eps, scale_shape, maxpool_kernel_size, clamp_min, clamp_max,
    BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    hw = tl.arange(0, BLOCK_H * BLOCK_W) + pid_hw * BLOCK_H * BLOCK_W
    h = hw // width
    w = hw % width
    mask = (pid_b < batch_size) & (pid_c < out_channels) & (hw < height * width)

    # Conv
    accum = tl.zeros([BLOCK_H * BLOCK_W], dtype=tl.float32)
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = h + kh - kernel_size // 2
                iw = w + kw - kernel_size // 2
                valid = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
                idx = ((pid_b * in_channels + ic) * height + ih) * width + iw
                x_val = tl.load(x_ptr + idx, mask=valid & mask, other=0.0)
                w_idx = (pid_c * in_channels + ic) * kernel_size * kernel_size + kh * kernel_size + kw
                w_val = tl.load(weight_ptr + w_idx)
                accum += x_val * w_val

    # Add bias
    b_val = tl.load(bias_ptr + pid_c)
    accum += b_val

    # GroupNorm
    group_size = out_channels // num_groups
    group = pid_c // group_size
    mean = tl.load(running_mean_ptr + pid_c)
    var = tl.load(running_var_ptr + pid_c)
    inv_std = tl.rsqrt(var + eps)
    norm_val = (accum - mean) * inv_std

    gn_w = tl.load(gn_weight_ptr + pid_c)
    gn_b = tl.load(gn_bias_ptr + pid_c)
    norm_val = norm_val * gn_w + gn_b

    # Scale
    scale = tl.load(scale_ptr + pid_c)
    norm_val *= scale

    # MaxPool
    out_h = height // maxpool_kernel_size
    out_w = width // maxpool_kernel_size
    pool_h = pid_hw // out_w
    pool_w = pid_hw % out_w
    max_val = -float('inf')
    for ph in range(maxpool_kernel_size):
        for pw in range(maxpool_kernel_size):
            ih = pool_h * maxpool_kernel_size + ph
            iw = pool_w * maxpool_kernel_size + pw
            valid = (ih < height) & (iw < width)
            idx = ((pid_b * out_channels + pid_c) * height + ih) * width + iw
            val = tl.load(x_ptr + idx, mask=valid & mask, other=-float('inf'))
            max_val = tl.maximum(max_val, val)

    # Clamp
    max_val = tl.clamp(max_val, clamp_min, clamp_max)

    out_idx = ((pid_b * out_channels + pid_c) * out_h + pool_h) * out_w + pool_w
    tl.store(out_ptr + out_idx, max_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Pre-compute running stats for GroupNorm
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

    def forward(self, x):
        batch_size, _, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        num_groups = self.group_norm.num_groups
        maxpool_kernel_size = self.maxpool.kernel_size
        out_h = height // maxpool_kernel_size
        out_w = width // maxpool_kernel_size

        # Allocate output
        out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

        # Grid dimensions
        BLOCK_C = 16
        BLOCK_H = 8
        BLOCK_W = 8
        grid = (batch_size, out_channels, (height * width + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W))

        # Launch fused kernel
        conv_gn_scale_max_clamp_kernel[grid](
            x, self.conv.weight, self.conv.bias,
            self.group_norm.weight, self.group_norm.bias, self.scale,
            out, self.running_mean, self.running_var,
            batch_size, self.conv.in_channels, out_channels, height, width, kernel_size,
            num_groups, self.group_norm.eps, self.scale.shape, maxpool_kernel_size,
            self.clamp_min, self.clamp_max,
            BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )
        return out