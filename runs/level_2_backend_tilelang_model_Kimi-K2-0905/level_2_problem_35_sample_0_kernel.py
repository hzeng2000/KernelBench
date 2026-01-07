import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv_sub_hardswish_maxpool_mish_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    pool_kernel_size: int,
    subtract_value: float,
    block_h: int = 8,
    block_w: int = 8,
    block_oc: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    pooled_height = out_height // pool_kernel_size
    pooled_width = out_width // pool_kernel_size

    @T.prim_func
    def fused_kernel(
        Input: T.Tensor((batch_size, in_channels, height, width), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        Output: T.Tensor((batch_size, out_channels, pooled_height, pooled_width), dtype),
    ):
        with T.Kernel(T.ceildiv(pooled_width, block_w), T.ceildiv(pooled_height, block_h), T.ceildiv(out_channels, block_oc), batch_size, threads=threads) as (bx, by, bz, bbatch):
            start_w = bx * block_w
            start_h = by * block_h
            start_oc = bz * block_oc
            start_n = bbatch

            # Allocate shared memory for input tile
            tile_h = block_h * pool_kernel_size + kernel_size - 1
            tile_w = block_w * pool_kernel_size + kernel_size - 1
            shared_input = T.alloc_shared((in_channels, tile_h, tile_w), dtype)
            # Allocate shared memory for weights
            shared_weight = T.alloc_shared((block_oc, in_channels, kernel_size, kernel_size), dtype)
            # Allocate local accumulator
            local_acc = T.alloc_fragment((block_oc, block_h * pool_kernel_size, block_w * pool_kernel_size), dtype)

            # Load weights to shared memory
            for o in T.Parallel(block_oc):
                for ic in range(in_channels):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            oc_idx = start_oc + o
                            if oc_idx < out_channels:
                                shared_weight[o, ic, kh, kw] = Weight[oc_idx, ic, kh, kw]

            # Load input tile to shared memory
            for ic in range(in_channels):
                for th in T.Parallel(tile_h):
                    for tw in range(tile_w):
                        h_idx = start_h * pool_kernel_size + th
                        w_idx = start_w * pool_kernel_size + tw
                        if h_idx < height and w_idx < width:
                            shared_input[ic, th, tw] = Input[start_n, ic, h_idx, w_idx]
                        else:
                            shared_input[ic, th, tw] = T.cast(0, dtype)

            # Compute convolution + subtract + hardswish + maxpool + mish
            for o in T.Parallel(block_oc):
                for ph in range(block_h * pool_kernel_size):
                    for pw in range(block_w * pool_kernel_size):
                        oc_idx = start_oc + o
                        if oc_idx < out_channels:
                            acc = T.cast(0, dtype)
                            for ic in range(in_channels):
                                for kh in range(kernel_size):
                                    for kw in range(kernel_size):
                                        acc += shared_weight[o, ic, kh, kw] * shared_input[ic, ph + kh, pw + kw]
                            # Subtract
                            acc = acc - T.cast(subtract_value, dtype)
                            # HardSwish
                            relu6 = T.min(T.max(acc + T.cast(3, dtype), T.cast(0, dtype)), T.cast(6, dtype))
                            hardswish = acc * relu6 / T.cast(6, dtype)
                            local_acc[o, ph, pw] = hardswish

            # MaxPool
            for o in T.Parallel(block_oc):
                for ph in range(block_h):
                    for pw in range(block_w):
                        oc_idx = start_oc + o
                        if oc_idx < out_channels:
                            max_val = T.cast(-1e10, dtype)
                            for kh in range(pool_kernel_size):
                                for kw in range(pool_kernel_size):
                                    val = local_acc[o, ph * pool_kernel_size + kh, pw * pool_kernel_size + kw]
                                    max_val = T.max(max_val, val)
                            # Mish activation
                            exp_val = T.exp(max_val)
                            mish_val = max_val * T.tanh(T.log(T.cast(1, dtype) + exp_val))
                            h_idx = start_h + ph
                            w_idx = start_w + pw
                            if h_idx < pooled_height and w_idx < pooled_width:
                                Output[start_n, oc_idx, h_idx, w_idx] = mish_val

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int, pool_kernel_size: int, subtract_value: float):
        key = (batch_size, in_channels, out_channels, height, width, kernel_size, pool_kernel_size, subtract_value)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_sub_hardswish_maxpool_mish_kernel(
                batch_size, in_channels, out_channels, height, width, kernel_size, pool_kernel_size, subtract_value
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get conv weights
        weight = self.conv.weight.half()
        
        batch_size, in_channels, height, width = x.shape
        out_channels = weight.shape[0]
        kernel_size = weight.shape[2]
        pool_kernel_size = self.pool_kernel_size
        
        # Calculate output dimensions
        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1
        pooled_height = out_height // pool_kernel_size
        pooled_width = out_width // pool_kernel_size
        
        # Get kernel
        kernel = self._get_kernel(batch_size, in_channels, out_channels, height, width, kernel_size, pool_kernel_size, self.subtract_value)
        
        # Allocate output tensor
        output = torch.empty(batch_size, out_channels, pooled_height, pooled_width, dtype=torch.float16, device=x.device)
        
        # Run kernel
        kernel(x.half(), weight, output)
        
        return output