import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv3d_gn_mean_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    D: int, H: int, W: int,
    kernel_size: int,
    num_groups: int,
    block_D: int = 4,
    block_H: int = 8,
    block_W: int = 8,
    block_C: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    pad = kernel_size // 2
    out_D = D
    out_H = H
    out_W = W
    
    # Precompute group norm parameters
    channels_per_group = out_channels // num_groups
    
    @T.prim_func
    def conv3d_gn_mean_kernel(
        Input: T.Tensor((batch_size, in_channels, D, H, W), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Gamma: T.Tensor((out_channels,), dtype),
        Beta: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size,), dtype),
    ):
        # Shared memory for reduction
        shared_sum = T.alloc_shared((threads,), dtype)
        shared_sum_sq = T.alloc_shared((threads,), dtype)
        
        with T.Kernel(T.ceildiv(out_W, block_W), T.ceildiv(out_H, block_H), T.ceildiv(out_D, block_D), batch_size, threads=threads) as (bx, by, bz, b_n):
            # Allocate local registers
            local_sum = T.alloc_local((1,), dtype)
            local_sum_sq = T.alloc_local((1,), dtype)
            local_count = T.alloc_local((1,), "int32")
            
            start_w = bx * block_W
            start_h = by * block_H
            start_d = bz * block_D
            
            local_sum[0] = T.cast(0.0, dtype)
            local_sum_sq[0] = T.cast(0.0, dtype)
            local_count[0] = 0
            
            # Process each output channel in blocks
            for c_out in T.Parallel(out_channels):
                # Compute group norm stats for this group
                group_id = c_out // channels_per_group
                group_start = group_id * channels_per_group
                group_end = min((group_id + 1) * channels_per_group, out_channels)
                
                # Compute conv3d for this channel
                for d in range(block_D):
                    out_d = start_d + d
                    if out_d >= out_D:
                        continue
                    for h in range(block_H):
                        out_h = start_h + h
                        if out_h >= out_H:
                            continue
                        for w in range(block_W):
                            out_w = start_w + w
                            if out_w >= out_W:
                                continue
                            
                            # Compute convolution at this position
                            conv_sum = T.cast(0.0, dtype)
                            for k_d in range(kernel_size):
                                in_d = out_d + k_d - pad
                                if in_d < 0 or in_d >= D:
                                    continue
                                for k_h in range(kernel_size):
                                    in_h = out_h + k_h - pad
                                    if in_h < 0 or in_h >= H:
                                        continue
                                    for k_w in range(kernel_size):
                                        in_w = out_w + k_w - pad
                                        if in_w < 0 or in_w >= W:
                                            continue
                                        for c_in in range(in_channels):
                                            conv_sum += Input[b_n, c_in, in_d, in_h, in_w] * Weight[c_out, c_in, k_d, k_h, k_w]
                            
                            # Add bias
                            conv_out = conv_sum + Bias[c_out]
                            
                            # Group norm computation
                            # Accumulate for mean/variance computation
                            local_sum[0] += conv_out
                            local_sum_sq[0] += conv_out * conv_out
                            local_count[0] += 1
            
            # Reduction within threadblock
            tid = T.thread_idx()
            shared_sum[tid] = local_sum[0]
            shared_sum_sq[tid] = local_sum_sq[0]
            T.tvm_storage_sync("shared")
            
            # Parallel reduction
            stride = threads // 2
            while stride > 0:
                if tid < stride:
                    shared_sum[tid] += shared_sum[tid + stride]
                    shared_sum_sq[tid] += shared_sum_sq[tid + stride]
                stride //= 2
                T.tvm_storage_sync("shared")
            
            # Final mean computation
            if tid == 0:
                total_elements = T.cast(batch_size * out_channels * out_D * out_H * out_W, dtype)
                mean_val = shared_sum[0] / total_elements
                Output[b_n] = mean_val

    return tilelang.compile(conv3d_gn_mean_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self._kernel_cache = {}
        
    def _get_kernel(self, batch_size: int, in_channels: int, out_channels: int, D: int, H: int, W: int, kernel_size: int, num_groups: int):
        key = (batch_size, in_channels, out_channels, D, H, W, kernel_size, num_groups)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv3d_gn_mean_kernel(
                batch_size, in_channels, out_channels, D, H, W, kernel_size, num_groups
            )
        return self._kernel_cache[key]

    def forward(self, x):
        batch_size, _, D, H, W = x.shape
        
        # Get kernel
        kernel = self._get_kernel(
            batch_size, self.conv.in_channels, self.conv.out_channels,
            D, H, W, self.conv.kernel_size[0], self.group_norm.num_groups
        )
        
        # Prepare inputs
        weight = self.conv.weight.half()
        bias = self.conv.bias.half()
        gamma = self.group_norm.weight.half()
        beta = self.group_norm.bias.half()
        
        # Run fused kernel
        output = kernel(x.half(), weight, bias, gamma, beta)
        
        return output