import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_kernel(
    batch: int, in_c: int, out_c: int, 
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    block_d: int = 2, block_h: int = 4, block_w: int = 4,
    threads: int = 128, dtype: str = "float16"
):
    
    @T.prim_func
    def conv_transpose3d_kernel(
        Input: T.Tensor((batch, in_c, in_d, in_h, in_w), dtype),
        Weight: T.Tensor((in_c, out_c, kernel_d, kernel_h, kernel_w), dtype),
        Output: T.Tensor((batch, out_c, out_d, out_h, out_w), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w, block_w), T.ceildiv(out_h, block_h), 
                     T.ceildiv(out_d, block_d), out_c, batch, threads=threads) as (bx, by, bz, co, b):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            
            for local_w, local_h, local_d in T.Parallel(block_w, block_h, block_d):
                ow = start_w + local_w
                oh = start_h + local_h
                od = start_d + local_d
                
                if ow < out_w and oh < out_h and od < out_d:
                    accum = T.alloc_fragment((1,), dtype, "local")
                    accum[0] = T.cast(0, dtype)
                    
                    for ic in range(in_c):
                        for kd in range(kernel_d):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    in_d_idx = od - kd * stride_d + pad_d
                                    in_h_idx = oh - kh * stride_h + pad_h
                                    in_w_idx = ow - kw * stride_w + pad_w
                                    
                                    if (in_d_idx >= 0 and in_d_idx < in_d and 
                                        in_h_idx >= 0 and in_h_idx < in_h and 
                                        in_w_idx >= 0 and in_w_idx < in_w):
                                        accum[0] += Input[b, ic, in_d_idx, in_h_idx, in_w_idx] * Weight[ic, co, kd, kh, kw]
                    
                    Output[b, co, od, oh, ow] = accum[0]

    return tilelang.compile(conv_transpose3d_kernel, out_idx=[2], target="cuda")


def build_scale_maxpool_gavg_kernel(
    batch: int, channels: int, 
    in_d: int, in_h: int, in_w: int,
    maxpool_kernel: int, scale: float,
    block_c: int = 8, threads: int = 128, dtype: str = "float16"
):
    
    @T.prim_func
    def scale_maxpool_gavg_kernel(
        Input: T.Tensor((batch, channels, in_d, in_h, in_w), dtype),
        Output: T.Tensor((batch, channels, 1, 1, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(channels, block_c), batch, threads=threads) as (bc, b):
            start_c = bc * block_c
            
            for local_c in T.Parallel(block_c):
                c = start_c + local_c
                if c < channels:
                    accum = T.alloc_fragment((1,), dtype, "local")
                    accum[0] = T.cast(0, dtype)
                    
                    # MaxPool3d with kernel_size=2, stride=2
                    out_d = in_d // 2
                    out_h = in_h // 2
                    out_w = in_w // 2
                    
                    for od in range(out_d):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                max_val = T.cast(-1e10, dtype)
                                for kd in range(maxpool_kernel):
                                    for kh in range(maxpool_kernel):
                                        for kw in range(maxpool_kernel):
                                            val = Input[b, c, od*2+kd, oh*2+kh, ow*2+kw]
                                            if val > max_val:
                                                max_val = val
                                accum[0] += max_val
                    
                    # Global average pooling and scale
                    total_elements = out_d * out_h * out_w
                    avg_val = accum[0] / T.cast(total_elements, dtype)
                    scaled_val = avg_val * T.cast(scale, dtype)
                    
                    # Clamp between 0 and 1
                    clamped_val = T.max(T.cast(0, dtype), T.min(scaled_val, T.cast(1, dtype)))
                    Output[b, c, 0, 0, 0] = clamped_val

    return tilelang.compile(scale_maxpool_gavg_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self._kernel_cache = {}
        
    def _get_conv_kernel(self, batch: int, in_c: int, out_c: int, 
                        in_d: int, in_h: int, in_w: int,
                        out_d: int, out_h: int, out_w: int,
                        kernel_d: int, kernel_h: int, kernel_w: int,
                        stride_d: int, stride_h: int, stride_w: int,
                        pad_d: int, pad_h: int, pad_w: int):
        key = ("conv", batch, in_c, out_c, in_d, in_h, in_w, out_d, out_h, out_w,
               kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_kernel(
                batch, in_c, out_c, in_d, in_h, in_w, out_d, out_h, out_w,
                kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w
            )
        return self._kernel_cache[key]
    
    def _get_pool_kernel(self, batch: int, channels: int, 
                        in_d: int, in_h: int, in_w: int,
                        maxpool_kernel: int, scale: float):
        key = ("pool", batch, channels, in_d, in_h, in_w, maxpool_kernel, scale)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_scale_maxpool_gavg_kernel(
                batch, channels, in_d, in_h, in_w, maxpool_kernel, scale
            )
        return self._kernel_cache[key]
    
    def forward(self, x):
        # ConvTranspose3d
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        
        # Calculate output dimensions
        out_d = (in_d - 1) * 2 - 2 * 1 + 3
        out_h = (in_h - 1) * 2 - 2 * 1 + 3
        out_w = (in_w - 1) * 2 - 2 * 1 + 3
        
        conv_kernel = self._get_conv_kernel(
            batch_size, in_channels, self.conv_transpose.out_channels,
            in_d, in_h, in_w, out_d, out_h, out_w,
            self.conv_transpose.kernel_size[0], self.conv_transpose.kernel_size[1], self.conv_transpose.kernel_size[2],
            self.conv_transpose.stride[0], self.conv_transpose.stride[1], self.conv_transpose.stride[2],
            self.conv_transpose.padding[0], self.conv_transpose.padding[1], self.conv_transpose.padding[2]
        )
        
        x_fp16 = x.half()
        weight_fp16 = self.conv_transpose.weight.half()
        conv_out = conv_kernel(x_fp16, weight_fp16)
        
        # Scale, MaxPool, GlobalAvgPool, Clamp - fused kernel
        pool_kernel = self._get_pool_kernel(
            batch_size, self.conv_transpose.out_channels,
            out_d, out_h, out_w, self.maxpool_kernel_size, self.scale
        )
        
        final_out = pool_kernel(conv_out)
        
        return final_out.float()