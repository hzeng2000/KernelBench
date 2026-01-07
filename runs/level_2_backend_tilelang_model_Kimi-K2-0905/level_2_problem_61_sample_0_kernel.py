import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose_relu_gn_kernel(
    batch_size: int,
    out_channels: int,
    out_d: int,
    out_h: int,
    out_w: int,
    in_channels: int,
    kernel_size: int,
    groups: int,
    block_d: int = 4,
    block_h: int = 4,
    block_w: int = 4,
    threads: int = 256,
    dtype: str = "float16"
):
    @T.prim_func
    def kernel(
        X: T.Tensor((batch_size, in_channels, out_d, out_h, out_w), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        Mean: T.Tensor((groups,), dtype),
        InvStd: T.Tensor((groups,), dtype),
        Gamma: T.Tensor((out_channels,), dtype),
        Beta: T.Tensor((out_channels,), dtype),
        Out: T.Tensor((batch_size, out_channels, out_d, out_h, out_w), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_w, block_w),
            T.ceildiv(out_h, block_h),
            T.ceildiv(out_d, block_d),
            out_channels,
            batch_size,
            threads=threads
        ) as (bx, by, bz, c, n):
            # Shared memory for intermediate results
            shared_mem = T.alloc_shared((block_d, block_h, block_w), dtype)
            
            # Local accumulators
            accum = T.alloc_fragment((block_d, block_h, block_w), dtype, 0.0)
            
            # Compute output position
            d_start = bz * block_d
            h_start = by * block_h
            w_start = bx * block_w
            
            # ConvTranspose3d computation (simplified - assuming stride=1, padding=0)
            for kd in range(kernel_size):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        for ic in range(in_channels):
                            for local_d in T.Parallel(block_d):
                                for local_h in T.Parallel(block_h):
                                    for local_w in T.Parallel(block_w):
                                        d = d_start + local_d
                                        h = h_start + local_h
                                        w = w_start + local_w
                                        
                                        if d < out_d and h < out_h and w < out_w:
                                            # Simplified conv transpose - assumes proper indexing
                                            accum[local_d, local_h, local_w] += (
                                                X[n, ic, d, h, w] * 
                                                Weight[ic, c, kd, kh, kw]
                                            )
            
            # Apply bias, ReLU, and group norm
            group_size = out_channels // groups
            group_idx = c // group_size
            
            for local_d in T.Parallel(block_d):
                for local_h in T.Parallel(block_h):
                    for local_w in T.Parallel(block_w):
                        d = d_start + local_d
                        h = h_start + local_h
                        w = w_start + local_w
                        
                        if d < out_d and h < out_h and w < out_w:
                            # Add bias
                            val = accum[local_d, local_h, local_w] + Bias[c]
                            
                            # ReLU
                            val = T.max(val, T.cast(0.0, dtype))
                            
                            # Group norm
                            normalized = (val - Mean[group_idx]) * InvStd[group_idx]
                            out_val = normalized * Gamma[c] + Beta[c]
                            
                            Out[n, c, d, h, w] = out_val
    
    return tilelang.compile(kernel, out_idx=[6], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        
        self._kernel_cache = {}
        self.groups = groups
        self.out_channels = out_channels
        
    def _get_kernel(self, batch_size: int, out_d: int, out_h: int, out_w: int, in_channels: int, kernel_size: int):
        key = (batch_size, out_d, out_h, out_w, in_channels, kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose_relu_gn_kernel(
                batch_size, self.out_channels, out_d, out_h, out_w, 
                in_channels, kernel_size, self.groups
            )
        return self._kernel_cache[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get output shape from conv transpose
        with torch.no_grad():
            dummy_out = self.conv_transpose(x)
            out_shape = dummy_out.shape
        
        batch_size, out_channels, out_d, out_h, out_w = out_shape
        in_channels = x.shape[1]
        
        # Prepare kernel inputs
        weight = self.conv_transpose.weight.half()  # [in_channels, out_channels, k, k, k]
        bias = self.conv_transpose.bias.half() if self.conv_transpose.bias is not None else torch.zeros(out_channels, dtype=torch.float16, device=x.device)
        
        # Compute group norm parameters
        with torch.no_grad():
            dummy_out = self.conv_transpose(x)
            dummy_out = self.relu(dummy_out)
            gn_out = self.group_norm(dummy_out)
            
            # Compute group statistics
            group_size = out_channels // self.groups
            mean = torch.zeros(self.groups, dtype=torch.float16, device=x.device)
            inv_std = torch.zeros(self.groups, dtype=torch.float16, device=x.device)
            
            for g in range(self.groups):
                start_c = g * group_size
                end_c = (g + 1) * group_size
                group_vals = dummy_out[:, start_c:end_c, :, :, :]
                mean[g] = group_vals.mean()
                inv_std[g] = 1.0 / (group_vals.std() + 1e-5)
        
        gamma = self.group_norm.weight.half()
        beta = self.group_norm.bias.half()
        
        # Get kernel
        kernel = self._get_kernel(batch_size, out_d, out_h, out_w, in_channels, self.conv_transpose.kernel_size[0])
        
        # Run fused kernel
        out = kernel(
            x.half().contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            mean.contiguous(),
            inv_std.contiguous(),
            gamma.contiguous(),
            beta.contiguous()
        )
        
        return out.view(out_shape)