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
    stride: int,
    padding: int,
    block_d: int = 4,
    block_h: int = 4,
    block_w: int = 4,
    threads: int = 256,
    dtype: str = "float16"
):
    out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size
    
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
            out_channels,
            batch_size,
            threads=threads
        ) as (bx, by, bz, oc, b):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            
            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                
                if d < out_depth and h < out_height and w < out_width:
                    acc = T.alloc_fragment((1,), dtype, "local")
                    acc[0] = T.cast(0.0, dtype)
                    
                    for ic in range(in_channels):
                        for kd in range(kernel_size):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    in_d = d + padding - kd
                                    in_h = h + padding - kh
                                    in_w = w + padding - kw
                                    
                                    if (in_d % stride == 0 and 
                                        in_h % stride == 0 and 
                                        in_w % stride == 0):
                                        in_d = in_d // stride
                                        in_h = in_h // stride
                                        in_w = in_w // stride
                                        
                                        if (in_d >= 0 and in_d < in_depth and
                                            in_h >= 0 and in_h < in_height and
                                            in_w >= 0 and in_w < in_width):
                                            acc[0] += Input[b, ic, in_d, in_h, in_w] * Weight[ic, oc, kd, kh, kw]
                    
                    Output[b, oc, d, h, w] = acc[0]
    
    return tilelang.compile(conv_transpose3d_kernel, out_idx=[2], target="cuda")


def build_fused_scale_pool_bias_scale_kernel(
    batch_size: int,
    channels: int,
    depth: int,
    height: int,
    width: int,
    pool_size: int = 2,
    block_c: int = 8,
    block_d: int = 4,
    block_h: int = 4,
    block_w: int = 4,
    threads: int = 256,
    dtype: str = "float16"
):
    out_depth = depth // pool_size
    out_height = height // pool_size
    out_width = width // pool_size
    
    @T.prim_func
    def fused_kernel(
        Input: T.Tensor((batch_size, channels, depth, height, width), dtype),
        Scale1: T.Tensor((1,), dtype),
        Bias: T.Tensor((channels, 1, 1, 1), dtype),
        Scale2: T.Tensor((1,), dtype),
        Output: T.Tensor((batch_size, channels, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_width, block_w),
            T.ceildiv(out_height, block_h),
            T.ceildiv(out_depth, block_d),
            T.ceildiv(channels, block_c),
            batch_size,
            threads=threads
        ) as (bx, by, bz, bc, b):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            start_c = bc * block_c
            
            for local_c, local_d, local_h, local_w in T.Parallel(block_c, block_d, block_h, block_w):
                c = start_c + local_c
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                
                if c < channels and d < out_depth and h < out_height and w < out_width:
                    pool_sum = T.alloc_fragment((1,), dtype, "local")
                    pool_sum[0] = T.cast(0.0, dtype)
                    
                    for pd in range(pool_size):
                        for ph in range(pool_size):
                            for pw in range(pool_size):
                                in_d = d * pool_size + pd
                                in_h = h * pool_size + ph
                                in_w = w * pool_size + pw
                                pool_sum[0] += Input[b, c, in_d, in_h, in_w]
                    
                    pooled = pool_sum[0] * T.cast(1.0 / (pool_size ** 3), dtype)
                    scaled1 = pooled * Scale1[0]
                    biased = scaled1 + Bias[c, 0, 0, 0]
                    scaled2 = biased * Scale2[0]
                    
                    Output[b, c, d, h, w] = scaled2
    
    return tilelang.compile(fused_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self._conv_kernel_cache = {}
        self._fused_kernel_cache = {}
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def _get_conv_kernel(self, batch_size: int, in_d: int, in_h: int, in_w: int):
        key = (batch_size, in_d, in_h, in_w)
        if key not in self._conv_kernel_cache:
            self._conv_kernel_cache[key] = build_conv_transpose3d_kernel(
                batch_size, self.in_channels, self.out_channels,
                in_d, in_h, in_w, self.kernel_size, self.stride, self.padding
            )
        return self._conv_kernel_cache[key]

    def _get_fused_kernel(self, batch_size: int, channels: int, depth: int, height: int, width: int):
        key = (batch_size, channels, depth, height, width)
        if key not in self._fused_kernel_cache:
            self._fused_kernel_cache[key] = build_fused_scale_pool_bias_scale_kernel(
                batch_size, channels, depth, height, width
            )
        return self._fused_kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().half()
        batch_size, _, in_d, in_h, in_w = x.shape
        
        conv_kernel = self._get_conv_kernel(batch_size, in_d, in_h, in_w)
        conv_weight = self.conv_transpose.weight.half().permute(1, 0, 2, 3, 4).contiguous()
        x = conv_kernel(x, conv_weight)
        
        _, _, out_d, out_h, out_w = x.shape
        fused_kernel = self._get_fused_kernel(batch_size, self.out_channels, out_d, out_h, out_w)
        
        scale1 = self.scale1.half().view(1)
        bias = self.bias.half()
        scale2 = self.scale2.half().view(1)
        
        x = fused_kernel(x, scale1, bias, scale2)
        
        return x.float()