import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_conv3d_hardswish_gn_mean_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gScale: cute.Tensor, gBias: cute.Tensor,
    gOut: cute.Tensor,
    batch_size: int, out_channels: int, out_d: int, out_h: int, out_w: int,
    in_channels: int, kernel_d: int, kernel_h: int, kernel_w: int,
    groups: int
):
    bidx, tidy, tidz = cute.arch.block_idx()
    tidx = cute.arch.thread_idx().x
    
    # Grid-strided loop over output spatial locations
    out_spatial_size = out_d * out_h * out_w
    total_threads = cute.arch.grid_dim().x * cute.arch.block_dim().x
    thread_id = bidx * cute.arch.block_dim().x + tidx
    
    for spatial_idx in range(thread_id, out_spatial_size, total_threads):
        od = spatial_idx // (out_h * out_w)
        oh = (spatial_idx // out_w) % out_h
        ow = spatial_idx % out_w
        
        # Compute group index
        group_size = out_channels // groups
        for group in range(groups):
            group_start = group * group_size
            group_end = (group + 1) * group_size
            
            # Compute mean for this group
            group_sum = 0.0
            group_count = 0
            
            for c in range(group_start, group_end):
                # Conv3D computation
                conv_val = 0.0
                for ic in range(in_channels):
                    for kd in range(kernel_d):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                id_coord = od + kd
                                ih_coord = oh + kh
                                iw_coord = ow + kw
                                if (id_coord < out_d + kernel_d - 1 and
                                    ih_coord < out_h + kernel_h - 1 and
                                    iw_coord < out_w + kernel_w - 1):
                                    x_val = gX[bidx, ic, id_coord, ih_coord, iw_coord]
                                    w_val = gW[c, ic, kd, kh, kw]
                                    conv_val += x_val * w_val
                
                # Add bias
                if gB.shape[0] > 0:
                    conv_val += gB[c]
                
                # HardSwish activation
                hardswish_val = conv_val * torch.min(torch.max(conv_val + 3.0, 0.0), 6.0) / 6.0
                
                # GroupNorm: compute group statistics
                group_sum += hardswish_val
                group_count += 1
                
                # Store intermediate for normalization
                if tidx == 0:  # Single thread per group for stats
                    group_mean = group_sum / group_count
                    
                    # Compute variance
                    group_var_sum = 0.0
                    for c2 in range(group_start, group_end):
                        # Recompute conv + hardswish for variance
                        conv_val2 = 0.0
                        for ic in range(in_channels):
                            for kd in range(kernel_d):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        id_coord = od + kd
                                        ih_coord = oh + kh
                                        iw_coord = ow + kw
                                        if (id_coord < out_d + kernel_d - 1 and
                                            ih_coord < out_h + kernel_h - 1 and
                                            iw_coord < out_w + kernel_w - 1):
                                            x_val = gX[bidx, ic, id_coord, ih_coord, iw_coord]
                                            w_val = gW[c2, ic, kd, kh, kw]
                                            conv_val2 += x_val * w_val
                        if gB.shape[0] > 0:
                            conv_val2 += gB[c2]
                        hardswish_val2 = conv_val2 * torch.min(torch.max(conv_val2 + 3.0, 0.0), 6.0)
                        group_var_sum += (hardswish_val2 - group_mean) ** 2
                    
                    group_var = group_var_sum / group_count
                    group_std = (group_var + 1e-5) ** 0.5
                    
                    # Normalize and apply scale/bias
                    for c3 in range(group_start, group_end):
                        # Recompute for final output
                        conv_val3 = 0.0
                        for ic in range(in_channels):
                            for kd in range(kernel_d):
                                for kh in range(kernel_h):
                                    for kw in range(kernel_w):
                                        id_coord = od + kd
                                        ih_coord = oh + kh
                                        iw_coord = ow + kw
                                        if (id_coord < out_d + kernel_d - 1 and
                                            ih_coord < out_h + kernel_h - 1 and
                                            iw_coord < out_w + kernel_w - 1):
                                            x_val = gX[bidx, ic, id_coord, ih_coord, iw_coord]
                                            w_val = gW[c3, ic, kd, kh, kw]
                                            conv_val3 += x_val * w_val
                        if gB.shape[0] > 0:
                            conv_val3 += gB[c3]
                        hardswish_val3 = conv_val3 * torch.min(torch.max(conv_val3 + 3.0, 0.0), 6.0)
                        
                        # GroupNorm
                        normalized = (hardswish_val3 - group_mean) / group_std
                        scaled = normalized * gScale[c3] + gBias[c3]
                        
                        # Accumulate for mean pooling
                        if c3 == group_start:  # Initialize output for this batch
                            gOut[bidx, c3] = 0.0
                        gOut[bidx, c3] += scaled / (out_d * out_h * out_w)

@cute.jit
def fused_conv3d_hardswish_gn_mean_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mScale: cute.Tensor, mBias: cute.Tensor,
    mOut: cute.Tensor,
    batch_size: int, out_channels: int, out_d: int, out_h: int, out_w: int,
    in_channels: int, kernel_d: int, kernel_h: int, kernel_w: int,
    groups: int
):
    threads_per_block = 256
    grid_x = (batch_size * out_d * out_h * out_w + threads_per_block - 1) // threads_per_block
    
    fused_conv3d_hardswish_gn_mean_kernel(
        mX, mW, mB, mScale, mBias, mOut,
        batch_size, out_channels, out_d, out_h, out_w,
        in_channels, kernel_d, kernel_h, kernel_w, groups
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        kernel_d, kernel_h, kernel_w = self.kernel_size
        
        # Compute output spatial dimensions (assuming stride=1, padding=0)
        out_d = D - kernel_d + 1
        out_h = H - kernel_h + 1
        out_w = W - kernel_w + 1
        
        # Get weights and biases
        weight = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else torch.empty(0, device=x.device)
        scale = self.group_norm.weight
        norm_bias = self.group_norm.bias
        
        # Prepare tensors
        x_contig = x.contiguous().cuda()
        weight_contig = weight.contiguous().cuda()
        bias_contig = bias.contiguous().cuda()
        scale_contig = scale.contiguous().cuda()
        norm_bias_contig = norm_bias.contiguous().cuda()
        out = torch.empty((B, weight.shape[0]), dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(weight_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mB = from_dlpack(bias_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mScale = from_dlpack(scale_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBias = from_dlpack(norm_bias_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile and launch kernel
        key = (x.dtype, weight.dtype)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_conv3d_hardswish_gn_mean_host,
                mX, mW, mB, mScale, mBias, mOut,
                B, weight.shape[0], out_d, out_h, out_w,
                C, kernel_d, kernel_h, kernel_w, self.num_groups
            )
            self.compiled[key] = compiled
        
        compiled(
            mX, mW, mB, mScale, mBias, mOut,
            B, weight.shape[0], out_d, out_h, out_w,
            C, kernel_d, kernel_h, kernel_w, self.num_groups
        )
        
        return out