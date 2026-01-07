import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_relu_group_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    running_mean_ptr, running_var_ptr, weight_gn_ptr, bias_gn_ptr,
    batch_size, in_channels, out_channels, D, H, W,
    kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w, out_d, out_h, out_w, groups,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_elements = batch_size * out_channels * out_d * out_h * out_w
    if pid * BLOCK_SIZE >= num_elements:
        return

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Compute output indices
    n = offsets // (out_channels * out_d * out_h * out_w)
    remainder = offsets % (out_channels * out_d * out_h * out_w)
    c_out = remainder // (out_d * out_h * out_w)
    remainder = remainder % (out_d * out_h * out_w)
    d_out = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    h_out = remainder // out_w
    w_out = remainder % out_w

    # Compute input region
    d_start = d_out * stride_d - pad_d
    h_start = h_out * stride_h - pad_h
    w_start = w_out * stride_w - pad_w

    acc = 0.0
    for c_in in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    d_in = d_start + kd
                    h_in = h_start + kh
                    w_in = w_start + kw
                    if d_in >= 0 and d_in < D and h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                        in_idx = n * in_channels * D * H * W + c_in * D * H * W + d_in * H * W + h_in * W + w_in
                        w_idx = c_out * in_channels * kernel_d * kernel_h * kernel_w + c_in * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw
                        acc += tl.load(input_ptr + in_idx) * tl.load(weight_ptr + w_idx)

    if bias_ptr:
        acc += tl.load(bias_ptr + c_out)

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # GroupNorm
    group_size = out_channels // groups
    group = c_out // group_size
    channels_per_group = out_channels // groups
    start_c = group * channels_per_group
    end_c = start_c + channels_per_group

    # Compute mean and var for this group
    sum_val = 0.0
    sum_sq = 0.0
    count = 0
    for c in range(start_c, end_c):
        for d in range(out_d):
            for h in range(out_h):
                for w in range(out_w):
                    idx = n * out_channels * out_d * out_h * out_w + c * out_d * out_h * out_w + d * out_h * out_w + h * out_w + w
                    val = tl.load(output_ptr + idx) if idx != offsets else acc
                    sum_val += val
                    sum_sq += val * val
                    count += 1

    mean = sum_val / count
    var = sum_sq / count - mean * mean
    std = tl.sqrt(var + 1e-5)

    # Normalize
    normalized = (acc - mean) / std

    # Scale and shift
    scale = tl.load(weight_gn_ptr + c_out)
    shift = tl.load(bias_gn_ptr + c_out)
    out_val = normalized * scale + shift

    tl.store(output_ptr + offsets, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        batch_size, in_channels, D, H, W = x.shape
        out_channels = self.conv_transpose.out_channels
        kernel_size = self.conv_transpose.kernel_size
        stride = self.conv_transpose.stride
        padding = self.conv_transpose.padding
        out_d = (D - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
        out_h = (H - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
        out_w = (W - 1) * stride[2] - 2 * padding[2] + kernel_size[2]

        output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

        num_elements = batch_size * out_channels * out_d * out_h * out_w
        BLOCK_SIZE = 128
        grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

        # Pre-compute mean and var for GroupNorm
        running_mean = torch.zeros(out_channels, device=x.device)
        running_var = torch.ones(out_channels, device=x.device)

        conv_transpose_relu_group_norm_kernel[grid](
            x, self.conv_transpose.weight, self.conv_transpose.bias, output,
            running_mean, running_var, self.group_norm.weight, self.group_norm.bias,
            batch_size, in_channels, out_channels, D, H, W,
            kernel_size[0], kernel_size[1], kernel_size[2], stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2], out_d, out_h, out_w, self.group_norm.num_groups,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output