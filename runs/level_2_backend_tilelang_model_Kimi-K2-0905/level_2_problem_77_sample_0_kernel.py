import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    in_depth: int,
    in_height: int,
    in_width: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    block_d: int = 4,
    block_h: int = 4,
    block_w: int = 4,
    block_out: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding
    
    @T.prim_func
    def conv_transpose3d_kernel(
        Input: T.Tensor((batch_size, in_channels, in_depth, in_height, in_width), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype),
        Output: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_w),
            T.ceildiv(out_height, block_h),
            T.ceildiv(out_depth, block_d),
            T.ceildiv(out_channels, block_out),
            batch_size,
            threads=threads
        ) as (bx, by, bz, bo, bi):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            start_o = bo * block_out
            
            for local_o, local_d, local_h, local_w in T.Parallel(block_out, block_d, block_h, block_w):
                o = start_o + local_o
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                
                if o < out_channels and d < out_depth and h < out_height and w < out_width:
                    acc = T.alloc_fragment((1,), dtype, scope="local")
                    acc[0] = T.cast(0.0, dtype)
                    
                    for ic in range(in_channels):
                        for kd in range(kernel_size):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    in_d = d + padding - kd
                                    in_h = h + padding - kh
                                    in_w = w + padding - kw
                                    
                                    if (in_d % stride == 0 and in_h % stride == 0 and in_w % stride == 0):
                                        in_d = in_d // stride
                                        in_h = in_h // stride
                                        in_w = in_w // stride
                                        
                                        if (in_d >= 0 and in_d < in_depth and
                                            in_h >= 0 and in_h < in_height and
                                            in_w >= 0 and in_w < in_width):
                                            acc[0] += Input[bi, ic, in_d, in_h, in_w] * Weight[ic, o, kd, kh, kw]
                    
                    Output[bi, o, d, h, w] = acc[0]
    
    return tilelang.compile(conv_transpose3d_kernel, out_idx=[2], target="cuda")


def build_fused_scale_bn_gpool_kernel(
    batch_size: int,
    channels: int,
    depth: int,
    height: int,
    width: int,
    scale_factor: float,
    eps: float,
    block_c: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def fused_kernel(
        Input: T.Tensor((batch_size, channels, depth, height, width), dtype),
        Scale: T.Tensor((channels,), dtype),
        Bias: T.Tensor((channels,), dtype),
        RunningMean: T.Tensor((channels,), dtype),
        RunningVar: T.Tensor((channels,), dtype),
        Output: T.Tensor((batch_size, channels, 1, 1, 1), dtype),
    ):
        with T.Kernel(
            T.ceildiv(channels, block_c),
            batch_size,
            threads=threads
        ) as (bc, bi):
            start_c = bc * block_c
            
            for local_c in T.Parallel(block_c):
                c = start_c + local_c
                
                if c < channels:
                    sum_val = T.alloc_fragment((1,), "float32", scope="local")
                    sum_val[0] = T.cast(0.0, "float32")
                    
                    for d in range(depth):
                        for h in range(height):
                            for w in range(width):
                                val = Input[bi, c, d, h, w]
                                scaled_val = val * T.cast(scale_factor, dtype)
                                bn_val = (scaled_val - RunningMean[c]) / T.sqrt(RunningVar[c] + T.cast(eps, dtype))
                                bn_val = bn_val * Scale[c] + Bias[c]
                                sum_val[0] += T.cast(bn_val, "float32")
                    
                    avg_val = sum_val[0] / T.cast(depth * height * width, "float32")
                    Output[bi, c, 0, 0, 0] = T.cast(avg_val, dtype)
    
    return tilelang.compile(fused_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self._kernel_cache = {}
        
    def _get_conv_kernel(self, batch_size: int, in_channels: int, out_channels: int, 
                        in_depth: int, in_height: int, in_width: int, kernel_size: int, tl_dtype: str):
        key = ("conv", batch_size, in_channels, out_channels, in_depth, in_height, in_width, kernel_size, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_kernel(
                batch_size, in_channels, out_channels, in_depth, in_height, in_width, kernel_size, dtype=tl_dtype
            )
        return self._kernel_cache[key]
    
    def _get_fused_kernel(self, batch_size: int, channels: int, depth: int, height: int, width: int, 
                         scale_factor: float, eps: float, tl_dtype: str):
        key = ("fused", batch_size, channels, depth, height, width, scale_factor, eps, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_scale_bn_gpool_kernel(
                batch_size, channels, depth, height, width, scale_factor, eps, dtype=tl_dtype
            )
        return self._kernel_cache[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvTranspose3d
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
        out_channels = self.conv_transpose.out_channels
        kernel_size = self.conv_transpose.kernel_size[0]
        
        x = x.contiguous().half()
        weight = self.conv_transpose.weight.contiguous().half()
        
        conv_kernel = self._get_conv_kernel(batch_size, in_channels, out_channels, 
                                          in_depth, in_height, in_width, kernel_size, "float16")
        x = conv_kernel(x, weight)
        
        # Fused scale + batch norm + global avg pool
        channels = out_channels
        depth, height, width = x.shape[2], x.shape[3], x.shape[4]
        
        scale = self.batch_norm.weight.contiguous().half()
        bias = self.batch_norm.bias.contiguous().half()
        running_mean = self.batch_norm.running_mean.contiguous().half()
        running_var = self.batch_norm.running_var.contiguous().half()
        
        fused_kernel = self._get_fused_kernel(batch_size, channels, depth, height, width, 
                                            self.scale_factor, self.batch_norm.eps, "float16")
        x = fused_kernel(x, scale, bias, running_mean, running_var)
        
        return x.float()