import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_relu_hardswish_kernel(
    batch: int, in_channels: int, out_channels: int, height: int, width: int,
    kernel_size: int, stride: int = 1, padding: int = 0,
    block_h: int = 8, block_w: int = 8, block_out: int = 32,
    threads: int = 256, dtype: str = "float16"
):
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    @T.prim_func
    def conv_relu_hardswish_kernel(
        Input: T.Tensor((batch, in_channels, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch, out_channels, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_w),
            T.ceildiv(out_height, block_h),
            T.ceildiv(out_channels, block_out),
            batch,
            threads=threads
        ) as (bx, by, bz, bbatch):
            # Allocate shared memory for input tile and weights
            shared_input = T.alloc_shared(
                (block_h + kernel_size - 1, block_w + kernel_size - 1, in_channels), dtype
            )
            shared_weight = T.alloc_shared((block_out, in_channels, kernel_size, kernel_size), dtype)

            # Allocate local accumulator
            local_acc = T.alloc_fragment((block_h, block_w, block_out), dtype, accum=True)

            # Initialize accumulators to zero
            for i, j, k in T.Parallel(block_h, block_w, block_out):
                local_acc[i, j, k] = T.cast(0, dtype)

            # Compute output tile start positions
            out_y_start = by * block_h
            out_x_start = bx * block_w
            out_c_start = bz * block_out

            # Load weights into shared memory
            for k, c, r, s in T.Parallel(block_out, in_channels, kernel_size, kernel_size):
                if out_c_start + k < out_channels and c < in_channels:
                    shared_weight[k, c, r, s] = Weight[out_c_start + k, c, r, s]

            # Loop over input channels for convolution
            for ic in range(in_channels):
                # Load input tile into shared memory
                for i, j in T.Parallel(block_h + kernel_size - 1, block_w + kernel_size - 1):
                    in_y = out_y_start * stride - padding + i
                    in_x = out_x_start * stride - padding + j
                    if in_y >= 0 and in_y < height and in_x >= 0 and in_x < width:
                        shared_input[i, j, ic] = Input[bbatch, ic, in_y, in_x]
                    else:
                        shared_input[i, j, ic] = T.cast(0, dtype)

                # Compute convolution for this input channel
                for i, j, k in T.Parallel(block_h, block_w, block_out):
                    for r in range(kernel_size):
                        for s in range(kernel_size):
                            in_y = i * stride + r
                            in_x = j * stride + s
                            if out_c_start + k < out_channels:
                                local_acc[i, j, k] += (
                                    shared_input[in_y, in_x, ic] * shared_weight[k, ic, r, s]
                                )

            # Apply bias, ReLU, and HardSwish
            for i, j, k in T.Parallel(block_h, block_w, block_out):
                out_y = out_y_start + i
                out_x = out_x_start + j
                out_c = out_c_start + k
                if out_y < out_height and out_x < out_width and out_c < out_channels:
                    # Add bias
                    val = local_acc[i, j, k] + Bias[out_c]
                    # ReLU
                    val = T.max(val, T.cast(0, dtype))
                    # HardSwish: x * clamp((x + 3)/6, 0, 1)
                    hardswish_val = val * T.clamp((val + T.cast(3, dtype)) / T.cast(6, dtype), T.cast(0, dtype), T.cast(1, dtype))
                    Output[bbatch, out_c, out_y, out_x] = hardswish_val

    return tilelang.compile(conv_relu_hardswish_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int):
        key = (batch, in_channels, out_channels, height, width, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_relu_hardswish_kernel(
                batch, in_channels, out_channels, height, width, kernel_size,
                stride=1, padding=0, block_h=8, block_w=8, block_out=32, threads=256, dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and in fp16
        x = x.contiguous().half()
        batch, _, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # Get kernel
        kernel = self._get_kernel(batch, x.shape[1], out_channels, height, width, kernel_size)

        # Prepare weight and bias in fp16
        weight = self.conv.weight.contiguous().half()
        bias = self.conv.bias.contiguous().half()

        # Allocate output tensor
        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1
        output = torch.empty(batch, out_channels, out_height, out_width, dtype=torch.float16, device=x.device)

        # Launch kernel
        kernel(x, weight, bias, output)

        return output