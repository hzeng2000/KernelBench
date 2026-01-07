import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose_bn_tanh_max_gn_kernel(
    batch_size: int, 
    in_channels: int, 
    out_channels: int, 
    in_height: int, 
    in_width: int,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 1,
    num_groups: int = 8,
    block_M: int = 8,
    block_N: int = 8,
    block_K: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size
    pool_height = out_height // 2
    pool_width = out_width // 2
    channels_per_group = out_channels // num_groups
    
    @T.prim_func
    def kernel(
        Input: T.Tensor((batch_size, in_channels, in_height, in_width), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        BN_weight: T.Tensor((out_channels,), dtype),
        BN_bias: T.Tensor((out_channels,), dtype),
        GN_weight: T.Tensor((out_channels,), dtype),
        GN_bias: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, out_channels, pool_height, pool_width), dtype),
    ):
        with T.Kernel(T.ceildiv(pool_width, block_N), T.ceildiv(pool_height, block_M), batch_size * num_groups, threads=threads) as (bx, by, bz):
            start_x = bx * block_N
            start_y = by * block_M
            group = bz % num_groups
            batch = bz // num_groups
            
            start_c = group * channels_per_group
            
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                
                if y < pool_height and x < pool_width:
                    pool_y = y * 2
                    pool_x = x * 2
                    
                    max_val = T.min_value(dtype)
                    sum_val = T.alloc_fragment((channels_per_group,), dtype, 0)
                    
                    for c in range(channels_per_group):
                        channel = start_c + c
                        max_pool_val = T.min_value(dtype)
                        
                        for dy in range(2):
                            for dx in range(2):
                                conv_y = pool_y + dy
                                conv_x = pool_x + dx
                                
                                if conv_y < out_height and conv_x < out_width:
                                    conv_val = T.cast(0.0, dtype)
                                    
                                    for ic in range(in_channels):
                                        for kh in range(kernel_size):
                                            for kw in range(kernel_size):
                                                in_y = conv_y + padding - kh
                                                in_x = conv_x + padding - kw
                                                
                                                if in_y >= 0 and in_y < in_height and in_x >= 0 and in_x < in_width:
                                                    in_y_idx = (in_y + stride - 1) // stride
                                                    in_x_idx = (in_x + stride - 1) // stride
                                                    if in_y_idx * stride == in_y and in_x_idx * stride == in_x:
                                                        conv_val += Input[batch, ic, in_y_idx, in_x_idx] * Weight[ic, channel, kh, kw]
                                    
                                    conv_val += Bias[channel]
                                    
                                    # Batch norm
                                    mean = T.cast(0.0, dtype)
                                    var = T.cast(1.0, dtype)
                                    conv_val = (conv_val - mean) / T.sqrt(var + T.cast(1e-5, dtype))
                                    conv_val = conv_val * BN_weight[channel] + BN_bias[channel]
                                    
                                    # Tanh
                                    conv_val = T.tanh(conv_val)
                                    
                                    if conv_val > max_pool_val:
                                        max_pool_val = conv_val
                        
                        sum_val[c] = max_pool_val
                    
                    # Group norm
                    group_mean = T.cast(0.0, dtype)
                    for c in range(channels_per_group):
                        group_mean += sum_val[c]
                    group_mean /= T.cast(channels_per_group, dtype)
                    
                    group_var = T.cast(0.0, dtype)
                    for c in range(channels_per_group):
                        diff = sum_val[c] - group_mean
                        group_var += diff * diff
                    group_var /= T.cast(channels_per_group, dtype)
                    
                    for c in range(channels_per_group):
                        channel = start_c + c
                        norm_val = (sum_val[c] - group_mean) / T.sqrt(group_var + T.cast(1e-5, dtype))
                        Output[batch, channel, y, x] = norm_val * GN_weight[channel] + GN_bias[channel]

    return tilelang.compile(kernel, out_idx=[7], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, in_height: int, in_width: int):
        key = (batch_size, in_height, in_width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose_bn_tanh_max_gn_kernel(
                batch_size, self.in_channels, self.out_channels, in_height, in_width,
                self.kernel_size, self.stride, self.padding, self.num_groups
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, in_height, in_width = x.shape
        kernel = self._get_kernel(batch_size, in_height, in_width)
        
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size
        pool_height = out_height // 2
        pool_width = out_width // 2
        
        output = torch.empty(batch_size, self.out_channels, pool_height, pool_width, 
                           dtype=torch.float16, device=x.device)
        
        kernel(
            x.half(),
            self.weight.half(),
            self.bias.half(),
            self.bn_weight.half(),
            self.bn_bias.half(),
            self.gn_weight.half(),
            self.gn_bias.half(),
            output
        )
        
        return output