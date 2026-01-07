import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    D, H, W,
    kernel_size,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_d, out_h, out_w,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)
    pid_od = tl.program_id(4)

    oc_start = pid_oc * BLOCK_OC
    od_start = pid_od * BLOCK_D
    oh_start = pid_oh * BLOCK_H
    ow_start = pid_ow * BLOCK_W

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for ic_start in range(0, in_channels, BLOCK_IC):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    id = od_start * stride_d - pad_d + kd
                    ih = oh_start * stride_h - pad_h + kh
                    iw = ow_start * stride_w - pad_w + kw

                    if id >= 0 and id < D and ih >= 0 and ih < H and iw >= 0 and iw < W:
                        for ic in range(BLOCK_IC):
                            if ic_start + ic < in_channels:
                                input_idx = (
                                    pid_b * in_channels * D * H * W +
                                    (ic_start + ic) * D * H * W +
                                    id * H * W +
                                    ih * W +
                                    iw
                                )
                                weight_idx = (
                                    (oc_start + tl.arange(0, BLOCK_OC)) * in_channels * kernel_size * kernel_size * kernel_size +
                                    (ic_start + ic) * kernel_size * kernel_size * kernel_size +
                                    kd * kernel_size * kernel_size +
                                    kh * kernel_size +
                                    kw
                                )
                                mask_oc = (oc_start + tl.arange(0, BLOCK_OC)) < out_channels
                                mask_ic = (ic_start + ic) < in_channels
                                input_val = tl.load(input_ptr + input_idx, mask=mask_ic, other=0.0)
                                weight_val = tl.load(weight_ptr + weight_idx, mask=mask_oc, other=0.0)
                                acc += input_val * weight_val

    if bias_ptr is not None:
        bias_idx = oc_start + tl.arange(0, BLOCK_OC)
        bias_val = tl.load(bias_ptr + bias_idx, mask=(oc_start + tl.arange(0, BLOCK_OC)) < out_channels, other=0.0)
        acc += bias_val

    for oc in range(BLOCK_OC):
        if oc_start + oc < out_channels:
            out_idx = (
                pid_b * out_channels * out_d * out_h * out_w +
                (oc_start + oc) * out_d * out_h * out_w +
                od_start * out_h * out_w +
                oh_start * out_w +
                ow_start
            )
            tl.store(output_ptr + out_idx, acc[oc])


@triton.jit
def group_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, num_channels, num_groups,
    D, H, W,
    eps,
    BLOCK_C: tl.constexpr, BLOCK_SPATIAL: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    channels_per_group = num_channels // num_groups
    c_start = pid_g * channels_per_group
    spatial_size = D * H * W

    mean = tl.zeros((BLOCK_C,), dtype=tl.float32)
    var = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for c in range(BLOCK_C):
        if c_start + c < num_channels:
            sum_val = 0.0
            count = 0
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        idx = pid_b * num_channels * D * H * W + (c_start + c) * D * H * W + d * H * W + h * W + w
                        val = tl.load(x_ptr + idx)
                        sum_val += val
                        count += 1
            mean[c] = sum_val / count

    for c in range(BLOCK_C):
        if c_start + c < num_channels:
            sum_sq = 0.0
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        idx = pid_b * num_channels * D * H * W + (c_start + c) * D * H * W + d * H * W + h * W + w
                        val = tl.load(x_ptr + idx)
                        sum_sq += (val - mean[c]) * (val - mean[c])
            var[c] = sum_sq / spatial_size

    for c in range(BLOCK_C):
        if c_start + c < num_channels:
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        idx = pid_b * num_channels * D * H * W + (c_start + c) * D * H * W + d * H * W + h * W + w
                        val = tl.load(x_ptr + idx)
                        norm_val = (val - mean[c]) / tl.sqrt(var[c] + eps)
                        if weight_ptr is not None:
                            weight_idx = c_start + c
                            weight_val = tl.load(weight_ptr + weight_idx)
                            norm_val *= weight_val
                        if bias_ptr is not None:
                            bias_idx = c_start + c
                            bias_val = tl.load(bias_ptr + bias_idx)
                            norm_val += bias_val
                        tl.store(out_ptr + idx, norm_val)


@triton.jit
def fused_conv3d_group_norm_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, gn_weight_ptr, gn_bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    D, H, W,
    kernel_size,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    out_d, out_h, out_w,
    num_groups,
    eps,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
    BLOCK_KD: tl.constexpr, BLOCK_KH: tl.constexpr, BLOCK_KW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_od = tl.program_id(2)
    pid_oh = tl.program_id(3)
    pid_ow = tl.program_id(4)

    oc_start = pid_oc * BLOCK_OC
    od_start = pid_od
    oh_start = pid_oh
    ow_start = pid_ow

    channels_per_group = out_channels // num_groups
    group_id = (oc_start // channels_per_group)

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for ic_start in range(0, in_channels, BLOCK_IC):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    id = od_start * stride_d - pad_d + kd
                    ih = oh_start * stride_h - pad_h + kh
                    iw = ow_start * stride_w - pad_w + kw

                    if id >= 0 and id < D and ih >= 0 and ih < H and iw >= 0 and iw < W:
                        for ic in range(BLOCK_IC):
                            if ic_start + ic < in_channels:
                                input_idx = (
                                    pid_b * in_channels * D * H * W +
                                    (ic_start + ic) * D * H * W +
                                    id * H * W +
                                    ih * W +
                                    iw
                                )
                                weight_idx = (
                                    (oc_start + tl.arange(0, BLOCK_OC)) * in_channels * kernel_size * kernel_size * kernel_size +
                                    (ic_start + ic) * kernel_size * kernel_size * kernel_size +
                                    kd * kernel_size * kernel_size +
                                    kh * kernel_size +
                                    kw
                                )
                                mask_oc = (oc_start + tl.arange(0, BLOCK_OC)) < out_channels
                                mask_ic = (ic_start + ic) < in_channels
                                input_val = tl.load(input_ptr + input_idx, mask=mask_ic, other=0.0)
                                weight_val = tl.load(weight_ptr + weight_idx, mask=mask_oc, other=0.0)
                                acc += input_val * weight_val

    if bias_ptr is not None:
        bias_idx = oc_start + tl.arange(0, BLOCK_OC)
        bias_val = tl.load(bias_ptr + bias_idx, mask=(oc_start + tl.arange(0, BLOCK_OC)) < out_channels, other=0.0)
        acc += bias_val

    spatial_size = out_d * out_h * out_w
    group_mean = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    group_var = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for oc in range(BLOCK_OC):
        if oc_start + oc < out_channels and (oc_start + oc) // channels_per_group == group_id:
            sum_val = acc[oc]
            mean_val = sum_val / spatial_size
            group_mean[oc] = mean_val

    for oc in range(BLOCK_OC):
        if oc_start + oc < out_channels and (oc_start + oc) // channels_per_group == group_id:
            var_val = (acc[oc] - group_mean[oc]) * (acc[oc] - group_mean[oc]) / spatial_size
            group_var[oc] = var_val

    for oc in range(BLOCK_OC):
        if oc_start + oc < out_channels:
            norm_val = (acc[oc] - group_mean[oc]) / tl.sqrt(group_var[oc] + eps)
            if gn_weight_ptr is not None:
                gn_weight_idx = oc_start + oc
                gn_weight_val = tl.load(gn_weight_ptr + gn_weight_idx, mask=(oc_start + tl.arange(0, BLOCK_OC)) < out_channels, other=1.0)
                norm_val *= gn_weight_val
            if gn_bias_ptr is not None:
                gn_bias_idx = oc_start + oc
                gn_bias_val = tl.load(gn_bias_ptr + gn_bias_idx, mask=(oc_start + tl.arange(0, BLOCK_OC)) < out_channels, other=0.0)
                norm_val += gn_bias_val
            out_idx = (
                pid_b * out_channels * out_d * out_h * out_w +
                (oc_start + oc) * out_d * out_h * out_w +
                od_start * out_h * out_w +
                oh_start * out_w +
                ow_start
            )
            tl.store(output_ptr + out_idx, norm_val)


def triton_conv3d_group_norm(x, weight, bias, gn_weight, gn_bias, num_groups, eps=1e-5):
    batch_size, in_channels, D, H, W = x.shape
    out_channels, _, kD, kH, kW = weight.shape
    stride = 1
    padding = 0
    out_d = D - kD + 1
    out_h = H - kH + 1
    out_w = W - kW + 1

    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

    BLOCK_OC = 8
    BLOCK_IC = 8
    grid = (batch_size, (out_channels + BLOCK_OC - 1) // BLOCK_OC, out_d, out_h, out_w)

    fused_conv3d_group_norm_relu_kernel[grid](
        x, weight, bias, gn_weight, gn_bias, output,
        batch_size, in_channels, out_channels,
        D, H, W,
        kD,
        stride, stride, stride,
        padding, padding, padding,
        out_d, out_h, out_w,
        num_groups,
        eps,
        BLOCK_OC=BLOCK_OC, BLOCK_IC=BLOCK_IC,
        BLOCK_KD=1, BLOCK_KH=1, BLOCK_KW=1
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups

    def forward(self, x):
        x = triton_conv3d_group_norm(x, self.conv.weight, self.conv.bias, self.group_norm.weight, self.group_norm.bias, self.num_groups)
        x = x.mean(dim=[1, 2, 3, 4])
        return x