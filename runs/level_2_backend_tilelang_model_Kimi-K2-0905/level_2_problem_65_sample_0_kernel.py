import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv_pool_sigmoid_sum_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    pool_kernel_size: int,
    block_M: int = 8,
    block_N: int = 32,
    block_K: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    pooled_height = out_height // pool_kernel_size
    pooled_width = out_width // pool_kernel_size
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_channels, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size,), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), threads=threads) as (bx,):
            batch_start = bx * block_M
            batch_idx = batch_start + T.thread_binding(0, block_M, thread="threadIdx.x")
            
            if batch_idx < batch_size:
                sum_val = T.alloc_fragment((1,), dtype, scope="local")
                sum_val[0] = T.cast(0.0, dtype)
                
                for out_c in T.Parallel(out_channels):
                    for ph in T.serial(pooled_height):
                        for pw in T.serial(pooled_width):
                            pool_sum = T.alloc_fragment((1,), dtype, scope="local")
                            pool_sum[0] = T.cast(0.0, dtype)
                            
                            for kh in T.serial(kernel_size):
                                for kw in T.serial(kernel_size):
                                    for ih in T.serial(pool_kernel_size):
                                        for iw in T.serial(pool_kernel_size):
                                            h = ph * pool_kernel_size + ih
                                            w = pw * pool_kernel_size + iw
                                            
                                            conv_sum = T.alloc_fragment((1,), dtype, scope="local")
                                            conv_sum[0] = T.cast(0.0, dtype)
                                            
                                            for in_c in T.serial(in_channels):
                                                conv_sum[0] += X[batch_idx, in_c, h + kh, w + kw] * W[out_c, in_c, kh, kw]
                                            
                                            conv_sum[0] += B[out_c]
                                            pool_sum[0] += conv_sum[0]
                            
                            pool_avg = pool_sum[0] / T.cast(pool_kernel_size * pool_kernel_size, dtype)
                            sigmoid_val = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-pool_avg))
                            sum_val[0] += sigmoid_val
                
                Output[batch_idx] = sum_val[0]

    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self._kernel_cache = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def _get_kernel(self, batch_size: int, height: int, width: int):
        key = (batch_size, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_pool_sigmoid_sum_kernel(
                batch_size=batch_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                height=height,
                width=width,
                kernel_size=self.kernel_size,
                pool_kernel_size=self.pool_kernel_size,
                dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        # Convert to fp16
        x_fp16 = x.half()
        self.conv.weight.data = self.conv.weight.data.half()
        self.conv.bias.data = self.conv.bias.data.half()
        
        kernel = self._get_kernel(batch_size, height, width)
        output = kernel(x_fp16, self.conv.weight, self.conv.bias)
        
        return output