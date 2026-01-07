import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv3d_kernel(
    batch: int, out_channels: int, in_channels: int, 
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int = 1, stride_h: int = 1, stride_w: int = 1,
    pad_d: int = 0, pad_h: int = 0, pad_w: int = 0,
    block_d: int = 4, block_h: int = 8, block_w: int = 8,
    threads: int = 256, dtype: str = "float16"
):
    
    @T.prim_func
    def fused_conv3d_kernel(
        Input: T.Tensor((batch, in_channels, out_d + 2*pad_d, out_h + 2*pad_h, out_w + 2*pad_w), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_d, kernel_h, kernel_w), dtype),
        Scale: T.Tensor((out_channels, 1, 1, 1), dtype),
        Bias: T.Tensor((out_channels, 1, 1, 1), dtype),
        Output: T.Tensor((batch, out_channels, out_d, out_h, out_w), dtype),
    ):
        with T.Kernel(T.ceildiv(out_w, block_w), T.ceildiv(out_h, block_h), T.ceildiv(out_d, block_d), out_channels, batch, threads=threads) as (bx, by, bz, oc, b):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            
            for local_d, local_h, local_w in T.Parallel(block_d, block_h, block_w):
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w
                
                if d < out_d and h < out_h and w < out_w:
                    acc = T.alloc_fragment((1,), "float32", 0.0)
                    
                    for ic in range(in_channels):
                        for kd in range(kernel_d):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    in_d = d * stride_d + kd - pad_d
                                    in_h = h * stride_h + kh - pad_h
                                    in_w = w * stride_w + kw - pad_w
                                    
                                    if in_d >= 0 and in_d < out_d + 2*pad_d and in_h >= 0 and in_h < out_h + 2*pad_h and in_w >= 0 and in_w < out_w + 2*pad_w:
                                        acc[0] += T.cast(Input[b, ic, in_d, in_h, in_w], "float32") * T.cast(Weight[oc, ic, kd, kh, kw], "float32")
                    
                    # Apply scaling, tanh, bias, sigmoid
                    val = acc[0]
                    val = val * T.cast(Scale[oc, 0, 0, 0], "float32")
                    val = T.tanh(val)
                    val = val * T.cast(Bias[oc, 0, 0, 0], "float32")
                    val = T.sigmoid(val)
                    
                    Output[b, oc, d, h, w] = T.cast(val, dtype)

    return tilelang.compile(fused_conv3d_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}
        
    def _get_kernel(self, batch: int, in_channels: int, out_channels: int, 
                   out_d: int, out_h: int, out_w: int, kernel_size: int):
        key = (batch, in_channels, out_channels, out_d, out_h, out_w, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv3d_kernel(
                batch, out_channels, in_channels, out_d, out_h, out_w,
                kernel_size, kernel_size, kernel_size, dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        batch, in_channels, in_d, in_h, in_w = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        pad = self.conv.padding[0]
        stride = self.conv.stride[0]
        
        out_d = (in_d + 2 * pad - kernel_size) // stride + 1
        out_h = (in_h + 2 * pad - kernel_size) // stride + 1
        out_w = (in_w + 2 * pad - kernel_size) // stride + 1
        
        kernel = self._get_kernel(batch, in_channels, out_channels, out_d, out_h, out_w, kernel_size)
        
        # Pad input
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad, pad, pad), mode='constant', value=0)
        
        # Get weight
        weight = self.conv.weight.half()
        
        # Get scale and bias
        scale = self.scaling_factor.half()
        bias = self.bias.half()
        
        # Run kernel
        output = kernel(x_padded, weight, scale, bias)
        
        return output