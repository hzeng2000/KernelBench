import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose_gelu_groupnorm_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int,
    stride: int,
    num_groups: int,
    block_M: int = 8,
    block_N: int = 32,
    block_K: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    out_height = (height - 1) * stride + kernel_size
    out_width = (width - 1) * stride + kernel_size
    
    @T.prim_func
    def kernel(
        Input: T.Tensor((batch_size, in_channels, height, width), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        RunningMean: T.Tensor((out_channels,), dtype),
        RunningVar: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, out_channels, out_height, out_width), dtype)
    ):
        # Precompute constants for GroupNorm
        channels_per_group = out_channels // num_groups
        eps = 1e-5
        
        with T.Kernel(T.ceildiv(out_width, block_N), T.ceildiv(out_height, block_M), batch_size * num_groups, threads=threads) as (bx, by, bz):
            # Compute output tile coordinates
            out_n = bz // num_groups
            group_id = bz % num_groups
            out_h_start = by * block_M
            out_w_start = bx * block_N
            
            # Allocate shared memory for weight tile
            weight_shared = T.alloc_shared((in_channels, channels_per_group, kernel_size, kernel_size), dtype)
            
            # Load weight tile for this group
            for ic in T.Parallel(in_channels):
                for oc in T.serial(channels_per_group):
                    for kh in T.serial(kernel_size):
                        for kw in T.serial(kernel_size):
                            global_oc = group_id * channels_per_group + oc
                            weight_shared[ic, oc, kh, kw] = Weight[ic, global_oc, kh, kw]
            
            # Allocate accumulator
            acc = T.alloc_fragment((block_M, block_N), "float32")
            
            # Compute output tile
            for out_h in T.serial(block_M):
                for out_w in T.serial(block_N):
                    h = out_h_start + out_h
                    w = out_w_start + out_w
                    
                    if h < out_height and w < out_width:
                        # Initialize accumulator
                        acc[out_h, out_w] = 0.0
                        
                        # Compute input coordinates
                        in_h = h - kernel_size + stride
                        in_w = w - kernel_size + stride
                        
                        # Accumulate convolution
                        for ic in T.serial(in_channels):
                            for kh in T.serial(kernel_size):
                                for kw in T.serial(kernel_size):
                                    in_h_coord = (in_h + kh) // stride
                                    in_w_coord = (in_w + kw) // stride
                                    
                                    if (in_h + kh) % stride == 0 and (in_w + kw) % stride == 0:
                                        if in_h_coord >= 0 and in_h_coord < height and in_w_coord >= 0 and in_w_coord < width:
                                            for oc in T.serial(channels_per_group):
                                                global_oc = group_id * channels_per_group + oc
                                                acc[out_h, out_w] += Input[out_n, ic, in_h_coord, in_w_coord] * weight_shared[ic, oc, kh, kw]
                        
                        # Apply bias
                        global_oc_base = group_id * channels_per_group
                        acc[out_h, out_w] += Bias[global_oc_base]
                        
                        # GELU activation
                        x = acc[out_h, out_w]
                        gelu_x = 0.5 * x * (1.0 + T.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
                        acc[out_h, out_w] = gelu_x
            
            # Compute group statistics for normalization
            group_mean = T.alloc_fragment((1,), "float32")
            group_var = T.alloc_fragment((1,), "float32")
            
            # Compute mean
            sum_val = 0.0
            count = 0
            for out_h in T.serial(block_M):
                for out_w in T.serial(block_N):
                    h = out_h_start + out_h
                    w = out_w_start + out_w
                    if h < out_height and w < out_width:
                        sum_val += acc[out_h, out_w]
                        count += 1
            
            if count > 0:
                group_mean[0] = sum_val / count
                
                # Compute variance
                sum_sq = 0.0
                for out_h in T.serial(block_M):
                    for out_w in T.serial(block_N):
                        h = out_h_start + out_h
                        w = out_w_start + out_w
                        if h < out_height and w < out_width:
                            diff = acc[out_h, out_w] - group_mean[0]
                            sum_sq += diff * diff
                
                group_var[0] = sum_sq / count
                
                # Apply GroupNorm
                for out_h in T.serial(block_M):
                    for out_w in T.serial(block_N):
                        h = out_h_start + out_h
                        w = out_w_start + out_w
                        if h < out_height and w < out_width:
                            normalized = (acc[out_h, out_w] - group_mean[0]) / T.sqrt(group_var[0] + eps)
                            # Apply running stats normalization
                            for oc in T.serial(channels_per_group):
                                global_oc = group_id * channels_per_group + oc
                                Output[out_n, global_oc, h, w] = normalized * RunningVar[global_oc] + RunningMean[global_oc]
    
    return tilelang.compile(kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self._kernel_cache = {}
        
    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, height: int, width: int, kernel_size: int, stride: int, num_groups: int):
        key = (batch_size, in_channels, out_channels, height, width, kernel_size, stride, num_groups)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose_gelu_groupnorm_kernel(
                batch_size, in_channels, out_channels, height, width, kernel_size, stride, num_groups
            )
        return self._kernel_cache[key]
    
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv_transpose.out_channels
        kernel_size = self.conv_transpose.kernel_size[0]
        stride = self.conv_transpose.stride[0]
        num_groups = self.group_norm.num_groups
        
        # Get kernel
        kernel = self._get_kernel(batch_size, in_channels, out_channels, height, width, kernel_size, stride, num_groups)
        
        # Prepare inputs
        weight = self.conv_transpose.weight.transpose(0, 1).contiguous().half()
        bias = self.conv_transpose.bias.half() if self.conv_transpose.bias is not None else torch.zeros(out_channels, device=x.device, dtype=torch.float16)
        
        # GroupNorm parameters
        running_mean = self.group_norm.weight.half()
        running_var = self.group_norm.bias.half()
        
        # Run kernel
        output = kernel(x.half(), weight, bias, running_mean, running_var)
        
        return output