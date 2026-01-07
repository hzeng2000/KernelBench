import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_fused_conv_transpose_maxpool_softmax_kernel(
    batch_size: int, out_channels: int, out_depth: int, out_height: int, out_width: int,
    pool_kernel_size: int, pool_stride: int, pool_padding: int,
    block_batch: int = 4, block_channel: int = 8, block_depth: int = 4,
    threads: int = 256, dtype: str = "float16"
):
    pooled_depth = (out_depth + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    pooled_height = (out_height + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    pooled_width = (out_width + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
        Subtract: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, pooled_depth, pooled_height, pooled_width), dtype)
    ):
        with T.Kernel(
            T.ceildiv(batch_size, block_batch),
            T.ceildiv(out_channels, block_channel),
            T.ceildiv(pooled_depth, block_depth),
            threads=threads
        ) as (b_blk, c_blk, d_blk):
            # Shared memory for softmax normalization
            shared_max = T.alloc_shared([block_batch, block_channel], dtype)
            shared_sum = T.alloc_shared([block_batch], dtype)
            
            for b in T.Parallel(block_batch):
                for pd in T.Parallel(block_depth):
                    for ph in T.Parallel(pooled_height):
                        for pw in T.Parallel(pooled_width):
                            # Max pooling first
                            max_val = T.min_value(dtype)
                            for kd in T.serial(pool_kernel_size):
                                for kh in T.serial(pool_kernel_size):
                                    for kw in T.serial(pool_kernel_size):
                                        od = d_blk * block_depth * pool_stride + pd * pool_stride + kd - pool_padding
                                        oh = ph * pool_stride + kh - pool_padding
                                        ow = pw * pool_stride + kw - pool_padding
                                        if (od >= 0 and od < out_depth and
                                            oh >= 0 and oh < out_height and
                                            ow >= 0 and ow < out_width):
                                            for c in T.serial(block_channel):
                                                global_c = c_blk * block_channel + c
                                                if global_c < out_channels:
                                                    val = X[
                                                        b_blk * block_batch + b,
                                                        global_c,
                                                        od, oh, ow
                                                    ]
                                                    max_val = T.max(max_val, val)
                            
                            # Subtract and compute softmax numerator
                            softmax_numer = T.alloc_local([block_channel], dtype)
                            for c in T.serial(block_channel):
                                global_c = c_blk * block_channel + c
                                if global_c < out_channels:
                                    pooled_val = max_val - Subtract[global_c]
                                    softmax_numer[c] = T.exp(pooled_val)
                                else:
                                    softmax_numer[c] = T.cast(0, dtype)
                            
                            # Compute softmax denominator (sum across channels)
                            sum_exp = T.cast(0, dtype)
                            for c in T.serial(block_channel):
                                sum_exp += softmax_numer[c]
                            
                            # Apply swish and max across channels
                            swish_max = T.cast(0, dtype)
                            for c in T.serial(block_channel):
                                global_c = c_blk * block_channel + c
                                if global_c < out_channels:
                                    softmax_val = softmax_numer[c] / (sum_exp + T.cast(1e-6, dtype))
                                    swish_val = T.sigmoid(softmax_val) * softmax_val
                                    swish_max = T.max(swish_max, swish_val)
                            
                            if (b_blk * block_batch + b < batch_size and
                                d_blk * block_depth + pd < pooled_depth):
                                Output[
                                    b_blk * block_batch + b,
                                    d_blk * block_depth + pd,
                                    ph, pw
                                ] = swish_max
    
    return tilelang.compile(kernel, out_idx=[2], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, out_channels: int, out_depth: int, out_height: int, out_width: int):
        key = (batch_size, out_channels, out_depth, out_height, out_width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_transpose_maxpool_softmax_kernel(
                batch_size, out_channels, out_depth, out_height, out_width,
                self.pool_kernel_size, self.pool_stride, self.pool_padding
            )
        return self._kernel_cache[key]

    def forward(self, x):
        # Compute conv transpose output shape
        conv_out = self.conv_transpose(x)
        batch_size, out_channels, out_depth, out_height, out_width = conv_out.shape
        
        # Get kernel and run fused operations
        kernel = self._get_kernel(batch_size, out_channels, out_depth, out_height, out_width)
        subtract_param = self.subtract.to(conv_out.dtype).contiguous()
        output = kernel(conv_out, subtract_param)
        
        return output