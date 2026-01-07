import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose3d_relu_kernel(
    batch_size: int, 
    out_channels: int, 
    out_depth: int, 
    out_height: int, 
    out_width: int,
    in_channels: int,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
    output_padding: int = 1,
    block_size: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    
    @T.prim_func
    def conv_transpose3d_relu_kernel(
        Input: T.Tensor((batch_size, in_channels, (out_depth - kernel_size + 2 * padding - output_padding) // stride + 1, 
                        (out_height - kernel_size + 2 * padding - output_padding) // stride + 1,
                        (out_width - kernel_size + 2 * padding - output_padding) // stride + 1), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype),
        Output: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_channels, block_size), T.ceildiv(out_width, block_size), 
                     T.ceildiv(out_height, block_size), T.ceildiv(out_depth, block_size), 
                     batch_size, threads=threads) as (oc_b, w_b, h_b, d_b, b):
            
            oc_start = oc_b * block_size
            w_start = w_b * block_size
            h_start = h_b * block_size
            d_start = d_b * block_size
            
            for local_oc, local_w, local_h, local_d in T.Parallel(block_size, block_size, block_size, block_size):
                oc = oc_start + local_oc
                w = w_start + local_w
                h = h_start + local_h
                d = d_start + local_d
                
                if oc < out_channels and w < out_width and h < out_height and d < out_depth:
                    acc = T.alloc_fragment((1,), dtype, scope="local")
                    acc[0] = T.cast(0.0, dtype)
                    
                    for ic in range(in_channels):
                        for kd in range(kernel_size):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    in_d = (d + padding - kd * stride - output_padding) // stride
                                    in_h = (h + padding - kh * stride - output_padding) // stride
                                    in_w = (w + padding - kw * stride - output_padding) // stride
                                    
                                    if (d + padding - kd * stride - output_padding) % stride == 0 and \
                                       (h + padding - kh * stride - output_padding) % stride == 0 and \
                                       (w + padding - kw * stride - output_padding) % stride == 0 and \
                                       in_d >= 0 and in_h >= 0 and in_w >= 0 and \
                                       in_d < Input.shape[2] and in_h < Input.shape[3] and in_w < Input.shape[4]:
                                        
                                        acc[0] += Input[b, ic, in_d, in_h, in_w] * Weight[ic, oc, kd, kh, kw]
                    
                    # Apply LeakyReLU
                    Output[b, oc, d, h, w] = T.max(acc[0], T.cast(0.0, dtype)) + T.cast(0.2, dtype) * T.min(acc[0], T.cast(0.0, dtype))
    
    return tilelang.compile(conv_transpose3d_relu_kernel, out_idx=[2], target="cuda")


def build_multiply_relu_maxpool_kernel(
    batch_size: int,
    channels: int,
    depth: int,
    height: int,
    width: int,
    multiplier_shape: tuple,
    block_size: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    
    @T.prim_func
    def multiply_relu_maxpool_kernel(
        Input: T.Tensor((batch_size, channels, depth, height, width), dtype),
        Multiplier: T.Tensor(multiplier_shape, dtype),
        Output: T.Tensor((batch_size, channels, depth // 2, height // 2, width // 2), dtype),
    ):
        with T.Kernel(T.ceildiv(width // 2, block_size), T.ceildiv(height // 2, block_size), 
                     T.ceildiv(depth // 2, block_size), channels, batch_size, threads=threads) as (w_b, h_b, d_b, c, b):
            
            w_start = w_b * block_size
            h_start = h_b * block_size
            d_start = d_b * block_size
            
            for local_w, local_h, local_d in T.Parallel(block_size, block_size, block_size):
                w = w_start + local_w
                h = h_start + local_h
                d = d_start + local_d
                
                if w < width // 2 and h < height // 2 and d < depth // 2 and c < channels:
                    max_val = T.alloc_fragment((1,), dtype, scope="local")
                    max_val[0] = T.cast(-1e10, dtype)
                    
                    # Max pooling 2x2x2
                    for kd in range(2):
                        for kh in range(2):
                            for kw in range(2):
                                in_d = d * 2 + kd
                                in_h = h * 2 + kh
                                in_w = w * 2 + kw
                                
                                if in_d < depth and in_h < height and in_w < width:
                                    # Multiply by learnable parameter and apply LeakyReLU
                                    val = Input[b, c, in_d, in_h, in_w] * Multiplier[c, 0, 0, 0]
                                    activated = T.max(val, T.cast(0.0, dtype)) + T.cast(0.2, dtype) * T.min(val, T.cast(0.0, dtype))
                                    max_val[0] = T.max(max_val[0], activated)
                    
                    Output[b, c, d, h, w] = max_val[0]
    
    return tilelang.compile(multiply_relu_maxpool_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                                stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        
        self._kernel_cache1 = {}
        self._kernel_cache2 = {}
        
    def _get_kernel1(self, batch_size: int, in_channels: int, out_channels: int, 
                     out_depth: int, out_height: int, out_width: int, tl_dtype: str):
        key = (batch_size, in_channels, out_channels, out_depth, out_height, out_width, tl_dtype)
        if key not in self._kernel_cache1:
            self._kernel_cache1[key] = build_conv_transpose3d_relu_kernel(
                batch_size, out_channels, out_depth, out_height, out_width, in_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1, dtype=tl_dtype
            )
        return self._kernel_cache1[key]
    
    def _get_kernel2(self, batch_size: int, channels: int, depth: int, height: int, 
                     width: int, multiplier_shape: tuple, tl_dtype: str):
        key = (batch_size, channels, depth, height, width, multiplier_shape, tl_dtype)
        if key not in self._kernel_cache2:
            self._kernel_cache2[key] = build_multiply_relu_maxpool_kernel(
                batch_size, channels, depth, height, width, multiplier_shape, dtype=tl_dtype
            )
        return self._kernel_cache2[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First kernel: ConvTranspose3d + LeakyReLU
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        
        # Compute output dimensions
        out_depth = (x.shape[2] - 1) * 2 - 2 * 1 + 1 * 3 + 1
        out_height = (x.shape[3] - 1) * 2 - 2 * 1 + 1 * 3 + 1
        out_width = (x.shape[4] - 1) * 2 - 2 * 1 + 1 * 3 + 1
        out_channels = self.conv_transpose.out_channels
        
        kernel1 = self._get_kernel1(batch_size, in_channels, out_channels, 
                                    out_depth, out_height, out_width, "float16")
        
        # Convert weight to proper format
        weight = self.conv_transpose.weight.permute(1, 0, 2, 3, 4).contiguous().half()
        
        x = kernel1(x.half(), weight)
        
        # Second kernel: Multiply + LeakyReLU + MaxPool
        kernel2 = self._get_kernel2(batch_size, out_channels, out_depth, out_height, 
                                    out_width, self.multiplier.shape, "float16")
        
        x = kernel2(x, self.multiplier.half())
        
        return x.float()