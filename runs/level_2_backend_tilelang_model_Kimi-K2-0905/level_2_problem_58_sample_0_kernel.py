import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_transpose_conv_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_d: int = 4,
    block_h: int = 8,
    block_w: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    out_depth = (depth - 1) * stride - 2 * padding + kernel_size
    out_height = (height - 1) * stride - 2 * padding + kernel_size
    out_width = (width - 1) * stride - 2 * padding + kernel_size

    @T.prim_func
    def fused_transpose_conv_kernel(
        Input: T.Tensor((batch_size, in_channels, depth, height, width), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((1, 1, 1, 1), dtype),
        Output: T.Tensor((batch_size, 1, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_w),
            T.ceildiv(out_height, block_h),
            T.ceildiv(out_depth, block_d),
            batch_size,
            threads=threads
        ) as (bx, by, bz, b):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d

            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w

                if d < out_depth and h < out_height and w < out_width:
                    max_val = T.min_value(dtype)
                    sum_exp = T.cast(0.0, dtype)
                    
                    # Compute conv transpose for each output channel
                    for oc in range(out_channels):
                        conv_val = T.cast(0.0, dtype)
                        for ic in range(in_channels):
                            for kd in range(kernel_size):
                                for kh in range(kernel_size):
                                    for kw in range(kernel_size):
                                        in_d = (d + padding - kd) // stride
                                        in_h = (h + padding - kh) // stride
                                        in_w = (w + padding - kw) // stride
                                        
                                        if (d + padding - kd) % stride == 0 and \
                                           (h + padding - kh) % stride == 0 and \
                                           (w + padding - kw) % stride == 0 and \
                                           in_d >= 0 and in_d < depth and \
                                           in_h >= 0 and in_h < height and \
                                           in_w >= 0 and in_w < width:
                                            conv_val += Input[b, ic, in_d, in_h, in_w] * Weight[ic, oc, kd, kh, kw]
                        
                        # LogSumExp computation
                        if oc == 0:
                            max_val = conv_val
                        else:
                            max_val = T.max(max_val, conv_val)
                    
                    # Second pass for LogSumExp
                    for oc in range(out_channels):
                        conv_val = T.cast(0.0, dtype)
                        for ic in range(in_channels):
                            for kd in range(kernel_size):
                                for kh in range(kernel_size):
                                    for kw in range(kernel_size):
                                        in_d = (d + padding - kd) // stride
                                        in_h = (h + padding - kh) // stride
                                        in_w = (w + padding - kw) // stride
                                        
                                        if (d + padding - kd) % stride == 0 and \
                                           (h + padding - kh) % stride == 0 and \
                                           (w + padding - kw) % stride == 0 and \
                                           in_d >= 0 and in_d < depth and \
                                           in_h >= 0 and in_h < height and \
                                           in_w >= 0 and in_w < width:
                                            conv_val += Input[b, ic, in_d, in_h, in_w] * Weight[ic, oc, kd, kh, kw]
                        
                        sum_exp += T.exp(conv_val - max_val)
                    
                    logsumexp_val = max_val + T.log(sum_exp)
                    
                    # HardSwish: x * sigmoid(x + 3) / 6
                    sigmoid_arg = logsumexp_val + T.cast(3.0, dtype)
                    sigmoid = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-sigmoid_arg))
                    hardswish_val = logsumexp_val * sigmoid / T.cast(6.0, dtype)
                    
                    # Subtract bias
                    biased_val = hardswish_val - Bias[0, 0, 0, 0]
                    
                    # Clamp between -1 and 1
                    clamped_val = T.max(T.cast(-1.0, dtype), T.min(T.cast(1.0, dtype), biased_val))
                    
                    Output[b, 0, d, h, w] = clamped_val

    return tilelang.compile(fused_transpose_conv_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self._kernel_cache = {}
        
    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, 
                   depth: int, height: int, width: int, kernel_size: int, 
                   stride: int, padding: int):
        key = (batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride, padding)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_transpose_conv_kernel(
                batch_size, in_channels, out_channels, depth, height, width,
                kernel_size, stride, padding
            )
        return self._kernel_cache[key]

    def forward(self, x):
        batch_size, in_channels, depth, height, width = x.shape
        out_channels = self.conv_transpose.out_channels
        kernel_size = self.conv_transpose.kernel_size[0]
        stride = self.conv_transpose.stride[0]
        padding = self.conv_transpose.padding[0]
        
        kernel = self._get_kernel(
            batch_size, in_channels, out_channels, depth, height, width,
            kernel_size, stride, padding
        )
        
        weight = self.conv_transpose.weight.transpose(0, 1).contiguous()
        output = kernel(x.contiguous(), weight, self.bias)
        
        return output