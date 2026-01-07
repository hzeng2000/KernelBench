import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv3d_mish_tanh_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    D: int,
    H: int,
    W: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    block_D: int = 4,
    block_H: int = 8,
    block_W: int = 8,
    block_out: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    out_D = (D + 2 * padding - kernel_size) // stride + 1
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1

    @T.prim_func
    def conv3d_mish_tanh_kernel(
        Input: T.Tensor((batch_size, in_channels, D, H, W), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, out_channels, out_D, out_H, out_W), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_W, block_W),
            T.ceildiv(out_H, block_H),
            T.ceildiv(out_D, block_D),
            T.ceildiv(out_channels, block_out),
            batch_size,
            threads=threads
        ) as (bx, by, bz, bo, b):
            # Allocate shared memory for input tile
            shared_input = T.alloc_shared((block_D * stride + kernel_size - 1, block_H * stride + kernel_size - 1, block_W * stride + kernel_size - 1, in_channels), dtype)
            # Allocate shared memory for weight tile
            shared_weight = T.alloc_shared((block_out, kernel_size, kernel_size, kernel_size, in_channels), dtype)
            # Allocate local accumulator
            local_acc = T.alloc_fragment((block_D, block_H, block_W, block_out), dtype, scope="local")
            # Allocate local input fragment
            local_input = T.alloc_fragment((kernel_size, kernel_size, kernel_size, in_channels), dtype, scope="local")
            # Allocate local weight fragment
            local_weight = T.alloc_fragment((kernel_size, kernel_size, kernel_size, in_channels), dtype, scope="local")

            # Initialize accumulators to zero
            for d, h, w, o in T.Parallel(block_D, block_H, block_W, block_out):
                local_acc[d, h, w, o] = T.cast(0, dtype)

            # Load weight tile into shared memory
            for o, kd, kh, kw, c in T.Parallel(block_out, kernel_size, kernel_size, kernel_size, in_channels):
                out_c = bo * block_out + o
                if out_c < out_channels:
                    shared_weight[o, kd, kh, kw, c] = Weight[out_c, c, kd, kh, kw]

            # Loop over input channels
            for c in T.serial(in_channels):
                # Load input tile into shared memory
                for d, h, w in T.Parallel(block_D, block_H, block_W):
                    out_d = bz * block_D + d
                    out_h = by * block_H + h
                    out_w = bx * block_W + w
                    if out_d < out_D and out_h < out_H and out_w < out_W:
                        for kd in T.serial(kernel_size):
                            for kh in T.serial(kernel_size):
                                for kw in T.serial(kernel_size):
                                    in_d = out_d * stride - padding + kd
                                    in_h = out_h * stride - padding + kh
                                    in_w = out_w * stride - padding + kw
                                    if 0 <= in_d < D and 0 <= in_h < H and 0 <= in_w < W:
                                        shared_input[d * stride + kd, h * stride + kh, w * stride + kw, c] = Input[b, c, in_d, in_h, in_w]
                                    else:
                                        shared_input[d * stride + kd, h * stride + kh, w * stride + kw, c] = T.cast(0, dtype)

                # Compute convolution
                for d, h, w, o in T.Parallel(block_D, block_H, block_W, block_out):
                    out_d = bz * block_D + d
                    out_h = by * block_H + h
                    out_w = bx * block_W + w
                    out_c = bo * block_out + o
                    if out_d < out_D and out_h < out_H and out_w < out_W and out_c < out_channels:
                        # Load input fragment
                        for kd, kh, kw in T.Parallel(kernel_size, kernel_size, kernel_size):
                            local_input[kd, kh, kw, c] = shared_input[d * stride + kd, h * stride + kh, w * stride + kw, c]
                        # Load weight fragment
                        for kd, kh, kw in T.Parallel(kernel_size, kernel_size, kernel_size):
                            local_weight[kd, kh, kw, c] = shared_weight[o, kd, kh, kw, c]
                        # Accumulate
                        for kd in T.serial(kernel_size):
                            for kh in T.serial(kernel_size):
                                for kw in T.serial(kernel_size):
                                    local_acc[d, h, w, o] += local_input[kd, kh, kw, c] * local_weight[kd, kh, kw, c]

            # Apply bias, Mish, and Tanh
            for d, h, w, o in T.Parallel(block_D, block_H, block_W, block_out):
                out_d = bz * block_D + d
                out_h = by * block_H + h
                out_w = bx * block_W + w
                out_c = bo * block_out + o
                if out_d < out_D and out_h < out_H and out_w < out_W and out_c < out_channels:
                    # Add bias
                    acc = local_acc[d, h, w, o] + Bias[out_c]
                    # Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
                    exp_acc = T.exp(acc)
                    softplus = T.log(T.cast(1, dtype) + exp_acc)
                    tanh_softplus = (T.exp(softplus) - T.exp(-softplus)) / (T.exp(softplus) + T.exp(-softplus))
                    mish_out = acc * tanh_softplus
                    # Tanh activation
                    tanh_out = (T.exp(mish_out) - T.exp(-mish_out)) / (T.exp(mish_out) + T.exp(-mish_out))
                    Output[b, out_c, out_d, out_h, out_w] = tanh_out

    return tilelang.compile(conv3d_mish_tanh_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._kernel_cache = {}
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, D: int, H: int, W: int):
        key = (batch_size, in_channels, out_channels, D, H, W, self.kernel_size, self.stride, self.padding)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv3d_mish_tanh_kernel(
                batch_size, in_channels, out_channels, D, H, W,
                self.kernel_size, self.stride, self.padding
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and in fp16
        x = x.contiguous().half()
        batch_size, in_channels, D, H, W = x.shape
        out_channels = self.conv.out_channels

        # Get kernel
        kernel = self._get_kernel(batch_size, in_channels, out_channels, D, H, W)

        # Get weight and bias
        weight = self.conv.weight.half()
        bias = self.conv.bias.half() if self.conv.bias is not None else torch.zeros(out_channels, dtype=torch.float16, device=x.device)

        # Run kernel
        output = kernel(x, weight, bias)

        return output