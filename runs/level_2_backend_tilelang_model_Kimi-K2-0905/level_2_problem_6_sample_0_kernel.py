import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv3d_softmax_kernel(
    batch_size: int,
    out_channels: int,
    out_depth: int,
    out_height: int,
    out_width: int,
    block_d: int = 4,
    block_h: int = 8,
    block_w: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def conv3d_softmax_kernel(
        Input: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
        Output: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_w), T.ceildiv(out_height, block_h), T.ceildiv(out_depth, block_d), batch_size, threads=threads) as (bx, by, bz, bb):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            start_b = bb

            # Shared memory for max and sum reduction
            max_shared = T.alloc_shared((block_d, block_h, block_w), dtype)
            sum_shared = T.alloc_shared((block_d, block_h, block_w), dtype)

            # Initialize max and sum
            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                if d < out_depth and h < out_height and w < out_width:
                    max_val = T.min_value(dtype)
                    # Find max across channels
                    for c in range(out_channels):
                        val = Input[start_b, c, d, h, w]
                        if val > max_val:
                            max_val = val
                    max_shared[local_d, local_h, local_w] = max_val

            T.sync_shared()

            # Compute exp and sum
            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                if d < out_depth and h < out_height and w < out_width:
                    max_val = max_shared[local_d, local_h, local_w]
                    sum_exp = T.cast(0.0, dtype)
                    # Compute exp and sum
                    for c in range(out_channels):
                        val = Input[start_b, c, d, h, w]
                        exp_val = T.exp(val - max_val)
                        sum_exp += exp_val
                    sum_shared[local_d, local_h, local_w] = sum_exp

            T.sync_shared()

            # Compute softmax
            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                if d < out_depth and h < out_height and w < out_width:
                    max_val = max_shared[local_d, local_h, local_w]
                    sum_val = sum_shared[local_d, local_h, local_w]
                    # Compute final softmax values
                    for c in range(out_channels):
                        val = Input[start_b, c, d, h, w]
                        exp_val = T.exp(val - max_val)
                        Output[start_b, c, d, h, w] = exp_val / sum_val

    return tilelang.compile(conv3d_softmax_kernel, out_idx=[1], target="cuda")


def build_maxpool3d_kernel(
    batch_size: int,
    channels: int,
    in_depth: int,
    in_height: int,
    in_width: int,
    pool_kernel_size: int,
    block_d: int = 4,
    block_h: int = 8,
    block_w: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    out_depth = in_depth // pool_kernel_size
    out_height = in_height // pool_kernel_size
    out_width = in_width // pool_kernel_size

    @T.prim_func
    def maxpool3d_kernel(
        Input: T.Tensor((batch_size, channels, in_depth, in_height, in_width), dtype),
        Output: T.Tensor((batch_size, channels, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_w), T.ceildiv(out_height, block_h), T.ceildiv(out_depth, block_d), batch_size, threads=threads) as (bx, by, bz, bb):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            start_b = bb

            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                if d < out_depth and h < out_height and w < out_width:
                    for c in range(channels):
                        max_val = T.min_value(dtype)
                        # Find max in pool window
                        for pd in range(pool_kernel_size):
                            for ph in range(pool_kernel_size):
                                for pw in range(pool_kernel_size):
                                    in_d = d * pool_kernel_size + pd
                                    in_h = h * pool_kernel_size + ph
                                    in_w = w * pool_kernel_size + pw
                                    val = Input[start_b, c, in_d, in_h, in_w]
                                    if val > max_val:
                                        max_val = val
                        Output[start_b, c, d, h, w] = max_val

    return tilelang.compile(maxpool3d_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self._conv_softmax_cache = {}
        self._pool1_cache = {}
        self._pool2_cache = {}

    def _get_conv_softmax_kernel(self, batch_size: int, out_channels: int, out_depth: int, out_height: int, out_width: int):
        key = (batch_size, out_channels, out_depth, out_height, out_width)
        if key not in self._conv_softmax_cache:
            self._conv_softmax_cache[key] = build_conv3d_softmax_kernel(batch_size, out_channels, out_depth, out_height, out_width)
        return self._conv_softmax_cache[key]

    def _get_pool_kernel(self, batch_size: int, channels: int, in_depth: int, in_height: int, in_width: int, pool_kernel_size: int):
        key = (batch_size, channels, in_depth, in_height, in_width, pool_kernel_size)
        if key not in self._pool1_cache:
            self._pool1_cache[key] = build_maxpool3d_kernel(batch_size, channels, in_depth, in_height, in_width, pool_kernel_size)
        return self._pool1_cache[key]

    def forward(self, x):
        # Conv3d
        x = self.conv(x)
        
        # Get conv output shape
        batch_size, out_channels, out_depth, out_height, out_width = x.shape
        
        # Convert to fp16
        x = x.half()
        
        # Fused Softmax
        kernel = self._get_conv_softmax_kernel(batch_size, out_channels, out_depth, out_height, out_width)
        x = kernel(x)
        
        # First MaxPool3d
        kernel1 = self._get_pool_kernel(batch_size, out_channels, out_depth, out_height, out_width, self.pool_kernel_size)
        x = kernel1(x)
        
        # Update dimensions after first pool
        out_depth //= self.pool_kernel_size
        out_height //= self.pool_kernel_size
        out_width //= self.pool_kernel_size
        
        # Second MaxPool3d
        kernel2 = self._get_pool_kernel(batch_size, out_channels, out_depth, out_height, out_width, self.pool_kernel_size)
        x = kernel2(x)
        
        return x.float()