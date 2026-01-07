import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_gelu_group_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    batch_size, in_channels, out_channels, height, width, kernel_size, stride,
    out_height, out_width, num_groups, eps,
    BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    hw = pid % (out_height * out_width)
    n = pid // (out_height * out_width)
    if n >= batch_size:
        return

    oh = hw // out_width
    ow = hw % out_width

    # Compute conv transpose
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = oh - kh
                iw = ow - kw
                if ih >= 0 and iw >= 0 and ih < height and iw < width:
                    x_idx = n * in_channels * height * width + ic * height * width + ih * width + iw
                    w_idx = ic * out_channels * kernel_size * kernel_size + tl.arange(0, BLOCK_SIZE) * kernel_size * kernel_size + kh * kernel_size + kw
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(w_ptr + w_idx, mask=tl.arange(0, BLOCK_SIZE) < out_channels, other=0.0)
                    acc += x_val * w_val

    # Add bias
    b_val = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < out_channels, other=0.0)
    acc += b_val

    # GELU
    gelu_out = 0.5 * acc * (1.0 + tl.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

    # GroupNorm
    channels_per_group = out_channels // num_groups
    group = tl.arange(0, BLOCK_SIZE) // channels_per_group
    group_mask = (tl.arange(0, BLOCK_SIZE) < out_channels) & (group == tl.program_id(1))

    # Compute group mean
    group_sum = tl.sum(gelu_out * group_mask)
    group_count = tl.sum(group_mask.to(tl.float32))
    group_mean = group_sum / tl.maximum(group_count, 1.0)

    # Compute group variance
    group_var_sum = tl.sum((gelu_out - group_mean) * (gelu_out - group_mean) * group_mask)
    group_var = group_var_sum / tl.maximum(group_count, 1.0)

    # Normalize
    normalized = (gelu_out - group_mean) / tl.sqrt(group_var + eps)

    # Apply weight and bias
    gn_weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < out_channels, other=1.0)
    gn_bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < out_channels, other=0.0)
    out = normalized * gn_weight + gn_bias

    # Store output
    out_idx = n * out_channels * out_height * out_width + tl.arange(0, BLOCK_SIZE) * out_height * out_width + oh * out_width + ow
    tl.store(out_ptr + out_idx, out, mask=(tl.arange(0, BLOCK_SIZE) < out_channels))


def triton_conv_transpose_gelu_group_norm(x, weight, bias, num_groups, weight_gn, bias_gn, eps=1e-5):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    stride = 1
    out_height = height
    out_width = width

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    weight_gn = weight_gn.contiguous()
    bias_gn = bias_gn.contiguous()

    out = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 64
    grid = (batch_size * out_height * out_width, num_groups)

    conv_transpose_gelu_group_norm_kernel[grid](
        x, weight, bias, out, None, None, weight_gn, bias_gn,
        batch_size, in_channels, out_channels, height, width, kernel_size, stride,
        out_height, out_width, num_groups, eps,
        BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=out_channels // num_groups
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        return triton_conv_transpose_gelu_group_norm(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.group_norm.num_groups,
            self.group_norm.weight,
            self.group_norm.bias
        )