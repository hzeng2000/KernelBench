import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_bn_scale_kernel(
    batch: int, in_channels: int, out_channels: int, height: int, width: int,
    kernel_size: int, stride: int = 1, padding: int = 0,
    block_h: int = 8, block_w: int = 8, block_k: int = 8,
    threads: int = 256, dtype: str = "float16"
):
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    @T.prim_func
    def conv_bn_scale_kernel(
        X: T.Tensor((batch, in_channels, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        running_mean: T.Tensor((out_channels,), "float32"),
        running_var: T.Tensor((out_channels,), "float32"),
        weight: T.Tensor((out_channels,), dtype),
        bias: T.Tensor((out_channels,), dtype),
        Scale: T.Tensor((1,), dtype),
        Y: T.Tensor((batch, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_w),
            T.ceildiv(out_height, block_h),
            T.ceildiv(out_channels, block_k),
            batch,
            threads=threads
        ) as (bx, by, bz, bn):
            start_x = bx * block_w
            start_y = by * block_h
            start_z = bz * block_k

            for local_y, local_x, local_z in T.Parallel(block_h, block_w, block_k):
                y = start_y + local_y
                x = start_x + local_x
                z = start_z + local_z

                if y < out_height and x < out_width and z < out_channels:
                    acc = 0.0
                    for k in range(in_channels):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                ih = y * stride + kh - padding
                                iw = x * stride + kw - padding
                                if 0 <= ih < height and 0 <= iw < width:
                                    acc += X[bn, k, ih, iw].astype("float32") * W[z, k, kh, kw].astype("float32")
                    acc += B[z].astype("float32")

                    # BatchNorm inference: (x - mean) / sqrt(var + eps) * gamma + beta
                    eps = 1e-5
                    bn_val = (acc - running_mean[z]) / T.sqrt(running_var[z] + eps)
                    bn_val = bn_val * weight[z].astype("float32") + bias[z].astype("float32")

                    # Scale
                    out_val = bn_val * Scale[0].astype("float32")

                    Y[bn, z, y, x] = out_val.astype(dtype)

    return tilelang.compile(conv_bn_scale_kernel, out_idx=[7], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, in_c: int, out_c: int, h: int, w: int, k: int, tl_dtype: str):
        key = (batch, in_c, out_c, h, w, k, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_bn_scale_kernel(
                batch, in_c, out_c, h, w, k, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure conv weight and bias are in fp16
        weight = self.conv.weight.half()
        bias = self.conv.bias.half() if self.conv.bias is not None else torch.zeros(self.conv.out_channels, device=x.device, dtype=torch.float16)

        # Get BN parameters
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        bn_weight = self.bn.weight.half()
        bn_bias = self.bn.bias.half()

        # Get input shape
        batch, in_c, h, w = x.shape
        out_c = self.conv.out_channels
        k = self.conv.kernel_size[0]

        # Pad input if needed
        pad = self.conv.padding[0]
        if pad > 0:
            x = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        # Build kernel
        kernel = self._get_kernel(batch, in_c, out_c, h, w, k, "float16")

        # Allocate output
        out_h = (h + 2 * pad - k) // self.conv.stride[0] + 1
        out_w = (w + 2 * pad - k) // self.conv.stride[0] + 1
        y = torch.empty(batch, out_c, out_h, out_w, device=x.device, dtype=torch.float16)

        # Run fused kernel
        kernel(
            x.half(),
            weight,
            bias,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            torch.tensor([self.scaling_factor], device=x.device, dtype=torch.float16),
            y
        )

        return y