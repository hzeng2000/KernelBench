import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_conv_transpose_swish_gn_hardswish_kernel(
    batch_size: int, out_channels: int, out_depth: int, out_height: int, out_width: int,
    in_channels: int, in_depth: int, in_height: int, in_width: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    groups: int, eps: float, block_size: int = 8, threads: int = 256
):
    assert out_channels % groups == 0
    channels_per_group = out_channels // groups

    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, in_channels, in_depth, in_height, in_width), "float16"),
        W: T.Tensor((in_channels, out_channels, kernel_d, kernel_h, kernel_w), "float16"),
        B: T.Tensor((out_channels,), "float16"),
        Mean: T.Tensor((batch_size, groups, 1, 1, 1), "float16"),
        Var: T.Tensor((batch_size, groups, 1, 1, 1), "float16"),
        Y: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), "float16"),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_size),
            T.ceildiv(out_height, block_size),
            T.ceildiv(out_depth, block_size),
            batch_size * out_channels,
            threads=threads,
        ) as (bx, by, bz, bc):
            for local_x, local_y, local_z in T.Parallel(block_size, block_size, block_size):
                x = bx * block_size + local_x
                y = by * block_size + local_y
                z = bz * block_size + local_z
                if x < out_width and y < out_height and z < out_depth:
                    b = bc // out_channels
                    oc = bc % out_channels
                    group_id = oc // channels_per_group
                    acc = T.cast(0.0, "float32")
                    for ic in range(in_channels):
                        for kd in range(kernel_d):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    in_z = z * stride_d - pad_d + kd
                                    in_y = y * stride_h - pad_h + kh
                                    in_x = x * stride_w - pad_w + kw
                                    if (
                                        0 <= in_z < in_depth
                                        and 0 <= in_y < in_height
                                        and 0 <= in_x < in_width
                                    ):
                                        acc += T.cast(
                                            X[b, ic, in_z, in_y, in_x], "float32"
                                        ) * T.cast(W[ic, oc, kd, kh, kw], "float32")
                    acc += T.cast(B[oc], "float32")
                    # Swish: x * sigmoid(x)
                    swish = acc / (1.0 + T.exp(-acc))
                    # Group norm placeholder (mean/var computed externally)
                    mean = T.cast(Mean[b, group_id, 0, 0, 0], "float32")
                    var = T.cast(Var[b, group_id, 0, 0, 0], "float32")
                    gn_out = (swish - mean) / T.sqrt(var + T.cast(eps, "float32"))
                    # HardSwish: x * relu6(x + 3) / 6
                    hardswish_in = gn_out
                    relu6 = T.min(T.max(hardswish_in + T.cast(3.0, "float32"), T.cast(0.0, "float32")), T.cast(6.0, "float32"))
                    out = hardswish_in * relu6 / T.cast(6.0, "float32")
                    Y[b, oc, z, y, x] = T.cast(out, "float16")

    return tilelang.compile(kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self._kernel_cache = {}
        self.groups = groups
        self.eps = eps

    def _get_kernel(self, key):
        if key not in self._kernel_cache:
            (
                batch_size, out_channels, out_depth, out_height, out_width,
                in_channels, in_depth, in_height, in_width,
                kernel_d, kernel_h, kernel_w,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                groups, eps,
            ) = key
            self._kernel_cache[key] = build_fused_conv_transpose_swish_gn_hardswish_kernel(
                batch_size, out_channels, out_depth, out_height, out_width,
                in_channels, in_depth, in_height, in_width,
                kernel_d, kernel_h, kernel_w,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                groups, eps,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output shape
        batch_size, in_channels, in_depth, in_height, in_width = x.shape
        stride_d, stride_h, stride_w = self.conv_transpose.stride
        pad_d, pad_h, pad_w = self.conv_transpose.padding
        kernel_d, kernel_h, kernel_w = self.conv_transpose.kernel_size
        out_channels = self.conv_transpose.out_channels
        out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d
        out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h
        out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w

        # Compute conv transpose output
        x_conv = self.conv_transpose(x)

        # Compute group norm stats
        x_swish = torch.sigmoid(x_conv) * x_conv
        x_reshaped = x_swish.view(batch_size, self.groups, -1)
        mean = x_reshaped.mean(dim=-1, keepdim=True).view(batch_size, self.groups, 1, 1, 1)
        var = x_reshaped.var(dim=-1, unbiased=False, keepdim=True).view(batch_size, self.groups, 1, 1, 1)

        # Build kernel key
        key = (
            batch_size, out_channels, out_depth, out_height, out_width,
            in_channels, in_depth, in_height, in_width,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            self.groups, self.eps,
        )
        kernel = self._get_kernel(key)

        # Run fused kernel
        W = self.conv_transpose.weight
        B = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros(out_channels, dtype=torch.float16, device=x.device)
        Y = kernel(x.half(), W.half(), B.half(), mean.half(), var.half())

        # Apply final HardSwish
        return torch.nn.functional.hardswish(Y)