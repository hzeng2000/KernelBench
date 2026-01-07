import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv3d_leakyrelu_add_clamp_gelu_kernel(
    batch_size: int,
    out_channels: int,
    out_depth: int,
    out_height: int,
    out_width: int,
    block_C: int = 16,
    block_D: int = 4,
    block_H: int = 8,
    block_W: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
        SumTensor: T.Tensor((out_channels, 1, 1, 1), dtype),
        Out: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_W),
            T.ceildiv(out_height, block_H),
            T.ceildiv(out_depth, block_D),
            T.ceildiv(out_channels, block_C),
            batch_size,
            threads=threads
        ) as (bx, by, bz, bc, bb):
            start_w = bx * block_W
            start_h = by * block_H
            start_d = bz * block_D
            start_c = bc * block_C
            start_b = bb

            # Allocate shared memory for SumTensor slice
            sum_shared = T.alloc_shared((block_C,), dtype)

            # Load SumTensor into shared memory
            for i in T.Parallel(block_C):
                c = start_c + i
                if c < out_channels:
                    sum_shared[i] = SumTensor[c, 0, 0, 0]

            # Process each element in the block
            for local_d, local_h, local_w in T.Parallel(block_D, block_H, block_W):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w

                if d < out_depth and h < out_height and w < out_width:
                    for local_c in range(block_C):
                        c = start_c + local_c
                        if c < out_channels:
                            # Load input value
                            val = X[start_b, c, d, h, w]

                            # Apply LeakyReLU
                            leaky_val = T.where(val > 0, val, val * T.cast(0.2, dtype))

                            # Add SumTensor
                            added = leaky_val + sum_shared[local_c]

                            # Clamp
                            clamped = T.max(T.min(added, T.cast(1.0, dtype)), T.cast(-1.0, dtype))

                            # Approximate GELU: x * sigmoid(1.702 * x)
                            gelu_approx = clamped * T.sigmoid(clamped * T.cast(1.702, dtype))

                            # Store result
                            Out[start_b, c, d, h, w] = gelu_approx

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, out_channels: int, out_depth: int, out_height: int, out_width: int):
        key = (batch_size, out_channels, out_depth, out_height, out_width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv3d_leakyrelu_add_clamp_gelu_kernel(
                batch_size, out_channels, out_depth, out_height, out_width
            )
        return self._kernel_cache[key]

    def forward(self, x):
        # Perform 3D convolution
        x = self.conv(x)

        # Get output dimensions
        batch_size, out_channels, out_depth, out_height, out_width = x.shape

        # Convert to FP16
        x = x.half()
        sum_tensor_fp16 = self.sum_tensor.half()

        # Get kernel
        kernel = self._get_kernel(batch_size, out_channels, out_depth, out_height, out_width)

        # Allocate output tensor
        out = torch.empty_like(x)

        # Launch fused kernel
        kernel(x, sum_tensor_fp16, out)

        return out