import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose_relu_kernel(
    batch: int, in_channels: int, out_channels: int, 
    in_h: int, in_w: int, out_h: int, out_w: int,
    kernel_h: int, kernel_w: int, stride: int, padding: int,
    block_M: int = 32, block_N: int = 32, block_K: int = 32,
    threads: int = 256, dtype: str = "float16"
):
    
    @T.prim_func
    def conv_transpose_kernel(
        Input: T.Tensor((batch, in_channels, in_h, in_w), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_h, kernel_w), dtype),
        Output: T.Tensor((batch, out_channels, out_h, out_w), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w, block_N), T.ceildiv(out_h, block_M), batch * out_channels, threads=threads) as (bx, by, bz):
            start_x = bx * block_N
            start_y = by * block_M
            batch_idx = bz // out_channels
            out_c = bz % out_channels

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < out_h and x < out_w:
                    acc = T.cast(0.0, dtype)
                    for in_c in range(in_channels):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                in_y = (y + padding - kh) // stride
                                in_x = (x + padding - kw) // stride
                                if (y + padding - kh) % stride == 0 and (x + padding - kw) % stride == 0:
                                    if in_y >= 0 and in_y < in_h and in_x >= 0 and in_x < in_w:
                                        acc += Input[batch_idx, in_c, in_y, in_x] * Weight[in_c, out_c, kh, kw]
                    Output[batch_idx, out_c, y, x] = acc

    return tilelang.compile(conv_transpose_kernel, out_idx=[2], target="cuda")


def build_maxpool_hardtanh_kernel(
    batch: int, channels: int, in_h: int, in_w: int,
    out_h: int, out_w: int, kernel_size: int, stride: int,
    min_val: float, max_val: float,
    block_M: int = 32, block_N: int = 32, threads: int = 256, dtype: str = "float16"
):
    
    @T.prim_func
    def maxpool_hardtanh_kernel(
        Input: T.Tensor((batch, channels, in_h, in_w), dtype),
        Output: T.Tensor((batch, channels, out_h, out_w), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w, block_N), T.ceildiv(out_h, block_M), batch * channels, threads=threads) as (bx, by, bz):
            start_x = bx * block_N
            start_y = by * block_M
            batch_idx = bz // channels
            c = bz % channels

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < out_h and x < out_w:
                    max_val_local = T.cast(-1e10, dtype)
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            in_y = y * stride + kh
                            in_x = x * stride + kw
                            if in_y < in_h and in_x < in_w:
                                val = Input[batch_idx, c, in_y, in_x]
                                if val > max_val_local:
                                    max_val_local = val
                    
                    # Apply hardtanh
                    if max_val_local < T.cast(min_val, dtype):
                        max_val_local = T.cast(min_val, dtype)
                    elif max_val_local > T.cast(max_val, dtype):
                        max_val_local = T.cast(max_val, dtype)
                    
                    Output[batch_idx, c, y, x] = max_val_local

    return tilelang.compile(maxpool_hardtanh_kernel, out_idx=[1], target="cuda")


def build_mean_tanh_kernel(
    batch: int, channels: int, h: int, w: int,
    threads: int = 256, dtype: str = "float16"
):
    
    @T.prim_func
    def mean_tanh_kernel(
        Input: T.Tensor((batch, channels, h, w), dtype),
        Output: T.Tensor((batch, channels, 1, 1), dtype),
    ):
        with T.Kernel(batch * channels, threads=threads) as bz:
            batch_idx = bz // channels
            c = bz % channels

            sum_val = T.cast(0.0, dtype)
            for y in range(h):
                for x in range(w):
                    sum_val += Input[batch_idx, c, y, x]
            
            mean_val = sum_val / T.cast(h * w, dtype)
            # tanh activation
            tanh_val = T.tanh(mean_val)
            Output[batch_idx, c, 0, 0] = tanh_val

    return tilelang.compile(mean_tanh_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._kernel_cache = {}

    def _get_conv_transpose_kernel(self, batch, in_h, in_w, out_h, out_w, tl_dtype):
        key = ("conv_transpose", batch, self.in_channels, self.out_channels, in_h, in_w, out_h, out_w, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose_relu_kernel(
                batch, self.in_channels, self.out_channels,
                in_h, in_w, out_h, out_w,
                self.kernel_size, self.kernel_size, self.stride, self.padding,
                dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def _get_maxpool_hardtanh_kernel(self, batch, channels, in_h, in_w, out_h, out_w, tl_dtype):
        key = ("maxpool_hardtanh", batch, channels, in_h, in_w, out_h, out_w, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_maxpool_hardtanh_kernel(
                batch, channels, in_h, in_w, out_h, out_w,
                self.maxpool_kernel_size, self.maxpool_stride,
                self.hardtanh_min, self.hardtanh_max,
                dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def _get_mean_tanh_kernel(self, batch, channels, h, w, tl_dtype):
        key = ("mean_tanh", batch, channels, h, w, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_mean_tanh_kernel(
                batch, channels, h, w, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def forward(self, x):
        # Get input dimensions
        batch, in_c, in_h, in_w = x.shape
        
        # Calculate output dimensions of conv_transpose
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        # Calculate output dimensions of maxpool
        pool_out_h = (out_h - self.maxpool_kernel_size) // self.maxpool_stride + 1
        pool_out_w = (out_w - self.maxpool_kernel_size) // self.maxpool_stride + 1
        
        # Convert to FP16
        x_fp16 = x.half()
        weight_fp16 = self.conv_transpose.weight.half()
        
        # ConvTranspose using TileLang kernel
        conv_transpose_kernel = self._get_conv_transpose_kernel(batch, in_h, in_w, out_h, out_w, "float16")
        conv_out = torch.empty(batch, self.out_channels, out_h, out_w, dtype=torch.float16, device=x.device)
        conv_transpose_kernel(x_fp16, weight_fp16, conv_out)
        
        # MaxPool + Hardtanh using TileLang kernel
        maxpool_hardtanh_kernel = self._get_maxpool_hardtanh_kernel(batch, self.out_channels, out_h, out_w, pool_out_h, pool_out_w, "float16")
        pool_out = torch.empty(batch, self.out_channels, pool_out_h, pool_out_w, dtype=torch.float16, device=x.device)
        maxpool_hardtanh_kernel(conv_out, pool_out)
        
        # Mean + Tanh using TileLang kernel
        mean_tanh_kernel = self._get_mean_tanh_kernel(batch, self.out_channels, pool_out_h, pool_out_w, "float16")
        final_out = torch.empty(batch, self.out_channels, 1, 1, dtype=torch.float16, device=x.device)
        mean_tanh_kernel(pool_out, final_out)
        
        return final_out.float()