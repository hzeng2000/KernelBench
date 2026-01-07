import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_conv_bn_tanh_kernel(
    batch: int,
    H: int,
    W: int,
    in_channels: int,
    out_channels: int,
    K: int,
    padding: int,
    H_out: int,
    W_out: int,
    dtype: str = "float16",
):
    @T.prim_func
    def fused_conv_bn_tanh_kernel(
        x: T.Tensor((batch, in_channels, H, W), dtype),
        weight: T.Tensor((in_channels, out_channels, K, K), dtype),
        running_mean: T.Tensor((out_channels,), dtype),
        running_var: T.Tensor((out_channels,), dtype),
        bn_weight: T.Tensor((out_channels,), dtype),
        bn_bias: T.Tensor((out_channels,), dtype),
        eps: T.float32,
        output: T.Tensor((batch, out_channels, H_out, W_out), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_channels, 8),
            T.ceildiv(H_out, 4),
            T.ceildiv(W_out, 4),
            threads=128,
        ) as (bx, by, bz):
            start_oc = bx * 8
            start_oh = by * 4
            start_ow = bz * 4
            for local_oc, local_oh, local_ow in T.Parallel(8, 4, 4):
                oc = start_oc + local_oc
                oh = start_oh + local_oh
                ow = start_ow + local_ow
                if oc < out_channels and oh < H_out and ow < W_out:
                    for b in T.serial(batch):
                        sum_val = T.cast(0.0, dtype)
                        for ic in T.serial(in_channels):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    ih = oh + kh - padding
                                    iw = ow + kw - padding
                                    if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                        sum_val += x[b, ic, ih, iw] * weight[ic, oc, kh, kw]
                        temp = (sum_val - running_mean[oc]) / T.sqrt(running_var[oc] + eps) * bn_weight[oc] + bn_bias[oc]
                        output[b, oc, oh, ow] = T.tanh(temp)

    return tilelang.compile(fused_conv_bn_tanh_kernel, out_idx=[7], target="cuda")


def build_group_norm_kernel(
    batch: int,
    channels: int,
    H: int,
    W: int,
    num_groups: int,
    dtype: str = "float16",
):
    num_channels_per_group = channels // num_groups

    @T.prim_func
    def group_norm_kernel(
        x: T.Tensor((batch, channels, H, W), dtype),
        gn_weight: T.Tensor((channels,), dtype),
        gn_bias: T.Tensor((channels,), dtype),
        eps: T.float32,
        output: T.Tensor((batch, channels, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(batch, 1), T.ceildiv(num_groups, 8), threads=128) as (bx, by):
            for local_b, local_g in T.Parallel(1, 8):
                b = bx * 1 + local_b
                g = by * 8 + local_g
                if b < batch and g < num_groups:
                    sum_val = T.reduce(
                        T.sum,
                        [c, h, w],
                        x[b, c, h, w],
                        init=T.cast(0.0, dtype),
                        where=(c >= g * num_channels_per_group and c < (g + 1) * num_channels_per_group and h >= 0 and h < H and w >= 0 and w < W),
                    )
                    sum_sq = T.reduce(
                        T.sum,
                        [c, h, w],
                        x[b, c, h, w] * x[b, c, h, w],
                        init=T.cast(0.0, dtype),
                        where=(c >= g * num_channels_per_group and c < (g + 1) * num_channels_per_group and h >= 0 and h < H and w >= 0 and w < W),
                    )
                    count = T.cast(num_channels_per_group * H * W, dtype)
                    mean = sum_val / count
                    var = sum_sq / count - mean * mean
                    for c in T.serial(num_channels_per_group):
                        for h in T.serial(H):
                            for w in T.serial(W):
                                c_idx = g * num_channels_per_group + c
                                output[b, c_idx, h, w] = (x[b, c_idx, h, w] - mean) / T.sqrt(var + eps) * gn_weight[c_idx] + gn_bias[c_idx]

    return tilelang.compile(group_norm_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.fused_kernel = build_fused_conv_bn_tanh_kernel(
            512, 32, 32, in_channels, out_channels, kernel_size, padding, 34, 34, "float16"
        )
        self.gn_kernel = build_group_norm_kernel(512, out_channels, 17, 17, num_groups, "float16")

    def forward(self, x):
        x = x.half()
        weight = self.conv_transpose.weight.half()
        running_mean = self.batch_norm.running_mean.half()
        running_var = self.batch_norm.running_var.half()
        bn_weight = self.batch_norm.weight.half()
        bn_bias = self.batch_norm.bias.half()
        eps = self.batch_norm.eps
        x = self.fused_kernel(x, weight, running_mean, running_var, bn_weight, bn_bias, eps)
        x = self.max_pool(x)
        gn_weight = self.group_norm.weight.half()
        gn_bias = self.group_norm.bias.half()
        eps_gn = self.group_norm.eps
        x = self.gn_kernel(x, gn_weight, gn_bias, eps_gn)
        return x