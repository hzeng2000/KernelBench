import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_bn_scale_maxpool_clamp_kernel(
    batch_size: int, in_channels: int, out_channels: int, height: int, width: int,
    kernel_size: int, num_groups: int, maxpool_kernel_size: int,
    block_H: int = 8, block_W: int = 8, block_K: int = 8, threads: int = 256,
    dtype: str = "float16"
):
    
    # Calculate output dimensions
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1
    pooled_height = out_height // maxpool_kernel_size
    pooled_width = out_width // maxpool_kernel_size
    
    channels_per_group = out_channels // num_groups
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_channels, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        scale: T.Tensor((out_channels, 1, 1), dtype),
        Y: T.Tensor((batch_size, out_channels, pooled_height, pooled_width), dtype),
        mean: T.Tensor((num_groups,), dtype),
        var: T.Tensor((num_groups,), dtype),
        gamma: T.Tensor((out_channels,), dtype),
        beta: T.Tensor((out_channels,), dtype),
    ):
        with T.Kernel(
            T.ceildiv(pooled_width, block_W),
            T.ceildiv(pooled_height, block_H),
            batch_size * num_groups,
            threads=threads
        ) as (bx, by, bz):
            
            # Thread-local buffers
            local_Y = T.alloc_buffer((block_H, block_W), dtype)
            local_max = T.alloc_buffer((block_H, block_W), dtype)
            
            batch_idx = bz // num_groups
            group_idx = bz % num_groups
            
            start_h = by * block_H * maxpool_kernel_size
            start_w = bx * block_W * maxpool_kernel_size
            
            for local_h in T.Parallel(block_H):
                for local_w in T.Parallel(block_W):
                    h_base = start_h + local_h * maxpool_kernel_size
                    w_base = start_w + local_w * maxpool_kernel_size
                    
                    # Initialize max pool value
                    max_val = T.min_value(dtype)
                    
                    for kh in range(maxpool_kernel_size):
                        for kw in range(maxpool_kernel_size):
                            h_in = h_base + kh
                            w_in = w_base + kw
                            
                            if h_in < out_height and w_in < out_width:
                                # Compute convolution for this position
                                conv_sum = T.alloc_buffer((1,), dtype)
                                conv_sum[0] = 0.0
                                
                                for oc in range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
                                    for ic in range(in_channels):
                                        for kh_conv in range(kernel_size):
                                            for kw_conv in range(kernel_size):
                                                h_conv = h_in + kh_conv
                                                w_conv = w_in + kw_conv
                                                if (h_conv < height and w_conv < width):
                                                    conv_sum[0] += X[batch_idx, ic, h_conv, w_conv] * W[oc, ic, kh_conv, kw_conv]
                                
                                # Add bias
                                conv_sum[0] += B[group_idx * channels_per_group]
                                
                                # Group normalization
                                group_mean = mean[group_idx]
                                group_var = var[group_idx]
                                normalized = (conv_sum[0] - group_mean) / T.sqrt(group_var + 1e-5)
                                
                                # Apply gamma and beta
                                oc_idx = group_idx * channels_per_group
                                normalized = normalized * gamma[oc_idx] + beta[oc_idx]
                                
                                # Scale
                                scaled = normalized * scale[oc_idx, 0, 0]
                                
                                # Clamp
                                clamped = T.max(T.min(scaled, dtype(clamp_max)), dtype(clamp_min))
                                
                                # Max pooling
                                max_val = T.max(max_val, clamped)
                    
                    local_max[local_h, local_w] = max_val
            
            # Write output
            for local_h in T.Parallel(block_H):
                for local_w in T.Parallel(block_W):
                    h_out = by * block_H + local_h
                    w_out = bx * block_W + local_w
                    
                    if (h_out < pooled_height and w_out < pooled_width and 
                        batch_idx < batch_size):
                        for oc in range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
                            Y[batch_idx, oc, h_out, w_out] = local_max[local_h, local_w]
    
    return tilelang.compile(fused_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, 
                 maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.kernel_cache = {}
        
    def _get_kernel(self, batch_size, in_channels, out_channels, height, width,
                     kernel_size, num_groups, maxpool_kernel_size):
        key = (batch_size, in_channels, out_channels, height, width,
               kernel_size, num_groups, maxpool_kernel_size)
        if key not in self.kernel_cache:
            self.kernel_cache[key] = build_conv_bn_scale_maxpool_clamp_kernel(
                batch_size, in_channels, out_channels, height, width,
                kernel_size, num_groups, maxpool_kernel_size
            )
        return self.kernel_cache[key]
    
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        num_groups = self.group_norm.num_groups
        maxpool_kernel_size = self.maxpool.kernel_size
        
        # Get kernel
        kernel = self._get_kernel(batch_size, in_channels, out_channels, height, width,
                                 kernel_size, num_groups, maxpool_kernel_size)
        
        # Prepare inputs
        x_c = x.contiguous().half()
        weight_c = self.conv.weight.contiguous().half()
        bias_c = self.conv.bias.contiguous().half()
        scale_c = self.scale.contiguous().half()
        
        # Compute group norm statistics
        conv_out = torch.nn.functional.conv2d(x_c, weight_c, bias_c)
        conv_out_reshaped = conv_out.view(batch_size, num_groups, -1)
        mean = conv_out_reshaped.mean(dim=-1, keepdim=False).half()
        var = conv_out_reshaped.var(dim=-1, keepdim=False, unbiased=False).half()
        
        # Gamma and beta from group norm
        gamma = self.group_norm.weight.contiguous().half()
        beta = self.group_norm.bias.contiguous().half()
        
        # Output tensor
        out_height = (height - kernel_size + 1) // maxpool_kernel_size
        out_width = (width - kernel_size + 1) // maxpool_kernel_size
        output = torch.empty(batch_size, out_channels, out_height, out_width, 
                           dtype=torch.float16, device=x.device)
        
        # Run kernel
        kernel(x_c, weight_c, bias_c, scale_c, output, mean, var, gamma, beta)
        
        return output.float()