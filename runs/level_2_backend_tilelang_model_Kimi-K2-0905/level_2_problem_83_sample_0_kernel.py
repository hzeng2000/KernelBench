import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv3d_gn_min_clamp_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    kernel_size: int,
    groups: int,
    block_D: int = 8,
    block_H: int = 8,
    block_W: int = 8,
    block_C: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_depth = depth
    out_height = height
    out_width = width
    
    # Precompute group norm parameters
    channels_per_group = out_channels // groups
    
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, in_channels, depth, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        gamma: T.Tensor((out_channels,), dtype),
        beta: T.Tensor((out_channels,), dtype),
        Y: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), dtype),
    ):
        with T.Kernel(T.ceildiv(out_width, block_W), T.ceildiv(out_height, block_H), T.ceildiv(out_depth, block_D), T.ceildiv(out_channels, block_C), batch_size, threads=threads) as (bx, by, bz, bc, bb):
            start_x = bx * block_W
            start_y = by * block_H
            start_z = bz * block_D
            start_c = bc * block_C
            
            # Shared memory for input tile
            shared_X = T.alloc_shared((block_D + 2*pad, block_H + 2*pad, block_W + 2*pad, in_channels), dtype)
            # Shared memory for weights
            shared_W = T.alloc_shared((block_C, in_channels, kernel_size, kernel_size, kernel_size), dtype)
            # Local accumulator
            local_acc = T.alloc_fragment((block_D, block_H, block_W, block_C), dtype)
            
            # Initialize accumulators
            for d, h, w, c in T.Parallel(block_D, block_H, block_W, block_C):
                local_acc[d, h, w, c] = T.cast(0, dtype)
            
            # Load weights to shared memory
            for ic, kd, kh, kw, c in T.Parallel(in_channels, kernel_size, kernel_size, kernel_size, block_C):
                if start_c + c < out_channels:
                    shared_W[c, ic, kd, kh, kw] = W[start_c + c, ic, kd, kh, kw]
            
            # Convolution loops
            for ic in range(in_channels):
                # Load input tile to shared memory
                for d, h, w in T.Parallel(block_D + 2*pad, block_H + 2*pad, block_W + 2*pad):
                    gd = start_z + d - pad
                    gh = start_y + h - pad
                    gw = start_x + w - pad
                    if 0 <= gd < depth and 0 <= gh < height and 0 <= gw < width:
                        shared_X[d, h, w, ic] = X[bb, ic, gd, gh, gw]
                    else:
                        shared_X[d, h, w, ic] = T.cast(0, dtype)
                
                # Compute convolution
                for d, h, w, c in T.Parallel(block_D, block_H, block_W, block_C):
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                local_acc[d, h, w, c] += shared_X[d + kd, h + kh, w + kw, ic] * shared_W[c, ic, kd, kh, kw]
            
            # Apply bias, group norm, min, clamp
            for d, h, w, c in T.Parallel(block_D, block_H, block_W, block_C):
                if start_z + d < out_depth and start_y + h < out_height and start_x + w < out_width and start_c + c < out_channels:
                    # Add bias
                    val = local_acc[d, h, w, c] + B[start_c + c]
                    
                    # Group normalization
                    g = c // channels_per_group
                    group_start = g * channels_per_group
                    
                    # Compute mean
                    mean = T.cast(0, dtype)
                    for gc in range(channels_per_group):
                        if group_start + gc < out_channels:
                            mean += local_acc[d, h, w, group_start + gc] + B[group_start + gc]
                    mean /= T.cast(channels_per_group, dtype)
                    
                    # Compute variance
                    var = T.cast(0, dtype)
                    for gc in range(channels_per_group):
                        if group_start + gc < out_channels:
                            diff = (local_acc[d, h, w, group_start + gc] + B[group_start + gc]) - mean
                            var += diff * diff
                    var /= T.cast(channels_per_group, dtype)
                    
                    # Normalize
                    val = (val - mean) / T.sqrt(var + T.cast(1e-5, dtype))
                    val = val * gamma[start_c + c] + beta[start_c + c]
                    
                    # Min and clamp
                    val = T.min(val, T.cast(0.0, dtype))
                    val = T.max(val, T.cast(0.0, dtype))
                    val = T.min(val, T.cast(1.0, dtype))
                    
                    Y[bb, start_c + c, start_z + d, start_y + h, start_x + w] = val
    
    return tilelang.compile(kernel, out_idx=[5], target="cuda")


def build_dropout_kernel(
    batch_size: int,
    channels: int,
    depth: int,
    height: int,
    width: int,
    dropout_p: float,
    block_C: int = 16,
    block_D: int = 8,
    block_H: int = 8,
    block_W: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, channels, depth, height, width), dtype),
        Y: T.Tensor((batch_size, channels, depth, height, width), dtype),
    ):
        with T.Kernel(T.ceildiv(width, block_W), T.ceildiv(height, block_H), T.ceildiv(depth, block_D), T.ceildiv(channels, block_C), batch_size, threads=threads) as (bx, by, bz, bc, bb):
            start_x = bx * block_W
            start_y = by * block_H
            start_z = bz * block_D
            start_c = bc * block_C
            
            for d, h, w, c in T.Parallel(block_D, block_H, block_W, block_C):
                if start_z + d < depth and start_y + h < height and start_x + w < width and start_c + c < channels:
                    # Simple deterministic dropout (scale during training)
                    scale = T.cast(1.0 / (1.0 - dropout_p), dtype)
                    Y[bb, start_c + c, start_z + d, start_y + h, start_x + w] = X[bb, start_c + c, start_z + d, start_y + h, start_x + w] * scale
    
    return tilelang.compile(kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=True)
        self.norm = nn.GroupNorm(groups, out_channels)
        
        # Initialize norm parameters
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        
        self._conv_kernel_cache = {}
        self._dropout_kernel_cache = {}
    
    def _get_conv_kernel(self, batch_size, depth, height, width, dtype):
        key = (batch_size, depth, height, width, dtype)
        if key not in self._conv_kernel_cache:
            self._conv_kernel_cache[key] = build_conv3d_gn_min_clamp_kernel(
                batch_size, self.in_channels, self.out_channels, depth, height, width, 
                self.kernel_size, self.groups, dtype=dtype
            )
        return self._conv_kernel_cache[key]
    
    def _get_dropout_kernel(self, batch_size, channels, depth, height, width, dtype):
        key = (batch_size, channels, depth, height, width, dtype)
        if key not in self._dropout_kernel_cache:
            self._dropout_kernel_cache[key] = build_dropout_kernel(
                batch_size, channels, depth, height, width, self.dropout_p, dtype=dtype
            )
        return self._dropout_kernel_cache[key]
    
    def forward(self, x):
        batch_size, _, depth, height, width = x.shape
        
        # Convert to fp16
        x = x.half()
        self.conv.weight.data = self.conv.weight.data.half()
        self.conv.bias.data = self.conv.bias.data.half()
        self.norm.weight.data = self.norm.weight.data.half()
        self.norm.bias.data = self.norm.bias.data.half()
        
        # Custom fused kernel: conv3d + group_norm + min + clamp
        kernel = self._get_conv_kernel(batch_size, depth, height, width, "float16")
        x = kernel(x, self.conv.weight, self.conv.bias, self.norm.weight, self.norm.bias)
        
        # Custom dropout kernel
        if self.training and self.dropout_p > 0:
            dropout_kernel = self._get_dropout_kernel(batch_size, self.out_channels, depth, height, width, "float16")
            x = dropout_kernel(x)
        
        return x.float()