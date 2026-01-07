import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv3d_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    D: int, H: int, W: int, kernel_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_d = bidx * bdimx + tidx
    out_h = bidy * bdimy + tidy
    out_w = bidz * bdimz + tidz
    
    out_D = D - kernel_size + 1
    out_H = H - kernel_size + 1
    out_W = W - kernel_size + 1
    
    if out_d < out_D and out_h < out_H and out_w < out_W:
        for b in range(batch_size):
            for o in range(out_channels):
                acc = 0.0
                for i in range(in_channels):
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                x_val = gX[b, i, out_d + kd, out_h + kh, out_w + kw]
                                w_val = gW[o, i, kd, kh, kw]
                                acc += x_val * w_val
                gY[b, o, out_d, out_h, out_w] = acc

@cute.kernel
def group_norm_kernel(
    gX: cute.Tensor, gY: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor,
    gGamma: cute.Tensor, gBeta: cute.Tensor,
    batch_size: int, num_groups: int, channels_per_group: int,
    D: int, H: int, W: int
):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    
    group_idx = bidx
    if group_idx >= batch_size * num_groups:
        return
    
    batch_idx = group_idx // num_groups
    group = group_idx % num_groups
    
    start_c = group * channels_per_group
    
    mean = 0.0
    count = 0
    
    for c in range(channels_per_group):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    val = gX[batch_idx, start_c + c, d, h, w]
                    mean += val
                    count += 1
    
    mean = mean / count
    gMean[group_idx] = mean
    
    var = 0.0
    for c in range(channels_per_group):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    val = gX[batch_idx, start_c + c, d, h, w]
                    diff = val - mean
                    var += diff * diff
    
    var = var / count
    gVar[group_idx] = var
    
    eps = 1e-5
    inv_std = 1.0 / math.sqrt(var + eps)
    
    for c in range(channels_per_group):
        gamma = gGamma[start_c + c]
        beta = gBeta[start_c + c]
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    val = gX[batch_idx, start_c + c, d, h, w]
                    norm_val = (val - mean) * inv_std
                    gY[batch_idx, start_c + c, d, h, w] = norm_val * gamma + beta

@cute.kernel
def fused_conv3d_group_norm_relu_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gY: cute.Tensor,
    gGamma: cute.Tensor, gBeta: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    D: int, H: int, W: int, kernel_size: int,
    num_groups: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_d = bidx * bdimx + tidx
    out_h = bidy * bdimy + tidy
    out_w = bidz * bdimz + tidz
    
    out_D = D - kernel_size + 1
    out_H = H - kernel_size + 1
    out_W = W - kernel_size + 1
    
    if out_d < out_D and out_h < out_H and out_w < out_W:
        channels_per_group = out_channels // num_groups
        
        for b in range(batch_size):
            for o in range(out_channels):
                acc = 0.0
                for i in range(in_channels):
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                x_val = gX[b, i, out_d + kd, out_h + kh, out_w + kw]
                                w_val = gW[o, i, kd, kh, kw]
                                acc += x_val * w_val
                
                group_idx = o // channels_per_group
                start_c = group_idx * channels_per_group
                
                mean = 0.0
                var = 0.0
                count = channels_per_group * out_D * out_H * out_W
                
                for c in range(channels_per_group):
                    for d in range(out_D):
                        for h in range(out_H):
                            for w in range(out_W):
                                val = acc if (c == o and d == out_d and h == out_h and w == out_w) else 0.0
                                mean += val
                
                mean = mean / count
                
                for c in range(channels_per_group):
                    for d in range(out_D):
                        for h in range(out_H):
                            for w in range(out_W):
                                val = acc if (c == o and d == out_d and h == out_h and w == out_w) else mean
                                diff = val - mean
                                var += diff * diff
                
                var = var / count
                eps = 1e-5
                inv_std = 1.0 / math.sqrt(var + eps)
                
                norm_val = (acc - mean) * inv_std
                gamma = gGamma[o]
                beta = gBeta[o]
                gY[b, o, out_d, out_h, out_w] = max(0.0, norm_val * gamma + beta)

@cute.jit
def fused_conv3d_group_norm_relu_host(
    mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor,
    mGamma: cute.Tensor, mBeta: cute.Tensor
):
    batch_size, in_channels, D, H, W = mX.shape
    out_channels, _, kernel_size, _, _ = mW.shape
    
    out_D = D - kernel_size + 1
    out_H = H - kernel_size + 1
    out_W = W - kernel_size + 1
    
    threads_per_block = 8
    grid_x = cute.ceil_div(out_D, threads_per_block)
    grid_y = cute.ceil_div(out_H, threads_per_block)
    grid_z = cute.ceil_div(out_W, threads_per_block)
    
    fused_conv3d_group_norm_relu_kernel(
        mX, mW, mY, mGamma, mBeta,
        batch_size, in_channels, out_channels,
        D, H, W, kernel_size, 8
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

@cute.jit
def reduce_mean_kernel(mY: cute.Tensor, mOut: cute.Tensor):
    batch_size = mY.shape[0]
    total_elements = 1
    for i in range(1, len(mY.shape)):
        total_elements *= mY.shape[i]
    
    threads_per_block = 256
    grid_x = cute.ceil_div(batch_size, threads_per_block)
    
    @cute.kernel
    def reduce_kernel(gY: cute.Tensor, gOut: cute.Tensor, total: int):
        tidx = cute.arch.thread_idx().x
        bidx = cute.arch.block_idx().x
        
        if bidx < gY.shape[0]:
            sum_val = 0.0
            for i in range(total):
                sum_val += gY[bidx, i]
            gOut[bidx] = sum_val / total
    
    reduce_kernel(mY, mOut, total_elements).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.num_groups = num_groups
        self.compiled = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, D, H, W = x.shape
        out_channels = self.weight.shape[0]
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        
        out_D = D - 3 + 1
        out_H = H - 3 + 1
        out_W = W - 3 + 1
        
        conv_out = torch.empty((batch_size, out_channels, out_D, out_H, out_W), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        if self.bias is not None:
            bias = self.bias.contiguous().cuda()
            conv_out += bias.view(1, -1, 1, 1, 1)
        
        norm_out = torch.empty_like(conv_out)
        mNormOut = from_dlpack(norm_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        channels_per_group = out_channels // self.num_groups
        group_mean = torch.empty((batch_size * self.num_groups,), dtype=x.dtype, device=x.device)
        group_var = torch.empty((batch_size * self.num_groups,), dtype=x.dtype, device=x.device)
        
        key = (x.dtype, batch_size, in_channels, out_channels, D, H, W)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_conv3d_group_norm_relu_host, mX, mW, mConvOut, 
                                  from_dlpack(self.group_norm.weight, assumed_align=16), 
                                  from_dlpack(self.group_norm.bias, assumed_align=16))
            self.compiled[key] = compiled
        
        compiled(mX, mW, mConvOut, 
                from_dlpack(self.group_norm.weight, assumed_align=16), 
                from_dlpack(self.group_norm.bias, assumed_align=16))
        
        mean_out = torch.empty((batch_size,), dtype=x.dtype, device=x.device)
        mMeanOut = from_dlpack(mean_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mFinalOut = from_dlpack(mConvOut, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        reduce_mean_kernel(mFinalOut, mMeanOut)
        
        return mean_out