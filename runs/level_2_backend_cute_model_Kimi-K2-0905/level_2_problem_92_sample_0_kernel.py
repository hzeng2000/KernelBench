import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_gn_tanh_hardswish_residual_logsumexp_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    gOut: cute.Tensor, gLogSumExp: cute.Tensor,
    N: int, C_out: int, H_out: int, W_out: int, groups: int, eps: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    hw = bidx * bdimx + tidx
    c = bidy * bdimy + tidy
    n = bidz * bdimz + tidz
    
    if n < N and c < C_out and hw < H_out * W_out:
        h = hw // W_out
        w = hw % W_out
        
        # Convolution
        sum_val = 0.0
        for kh in range(3):
            for kw in range(3):
                for ci in range(gX.shape[1]):
                    h_in = h + kh - 1
                    w_in = w + kw - 1
                    if h_in >= 0 and h_in < gX.shape[2] and w_in >= 0 and w_in < gX.shape[3]:
                        sum_val += gX[n, ci, h_in, w_in] * gW[c, ci, kh, kw]
        
        conv_val = sum_val + gB[c]
        
        # Group Normalization
        group_size = C_out // groups
        group = c // group_size
        mean = gMean[n, group]
        var = gVar[n, group]
        inv_std = 1.0 / (var + eps).sqrt()
        norm_val = (conv_val - mean) * inv_std
        norm_val = norm_val * gGamma[c] + gBeta[c]
        
        # Tanh
        tanh_val = (1.0 - (-2.0 * norm_val).exp()) / (1.0 + (-2.0 * norm_val).exp())
        
        # HardSwish
        hard_swish_val = tanh_val * max(0.0, min(1.0, (tanh_val + 3.0) / 6.0))
        
        # Residual Addition
        residual_val = conv_val + hard_swish_val
        
        gOut[n, c, h, w] = residual_val
        
        # LogSumExp reduction (partial)
        cute.atomic_add(gLogSumExp[n, 0, h, w], residual_val.exp())

@cute.jit
def fused_conv_gn_tanh_hardswish_residual_logsumexp_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    mOut: cute.Tensor, mLogSumExp: cute.Tensor,
    N: int, C_out: int, H_out: int, W_out: int, groups: int, eps: float
):
    threads_x = 16
    threads_y = 16
    threads_z = 4
    
    blocks_x = cute.ceil_div(H_out * W_out, threads_x)
    blocks_y = cute.ceil_div(C_out, threads_y)
    blocks_z = cute.ceil_div(N, threads_z)
    
    conv_gn_tanh_hardswish_residual_logsumexp_kernel(
        mX, mW, mB, mMean, mVar, mGamma, mBeta, mOut, mLogSumExp,
        N, C_out, H_out, W_out, groups, eps
    ).launch(
        grid=(blocks_x, blocks_y, blocks_z),
        block=(threads_x, threads_y, threads_z)
    )

@cute.kernel
def compute_group_stats_kernel(
    gConv: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor,
    N: int, C_out: int, H_out: int, W_out: int, groups: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    hw = bidx * bdimx + tidx
    group = bidy * bdimy + tidy
    n = bidz * bdimz + tidz
    
    if n < N and group < groups and hw < H_out * W_out:
        h = hw // W_out
        w = hw % W_out
        
        group_size = C_out // groups
        start_c = group * group_size
        end_c = start_c + group_size
        
        # Compute mean
        sum_val = 0.0
        for c in range(start_c, end_c):
            sum_val += gConv[n, c, h, w]
        mean = sum_val / group_size
        gMean[n, group, h, w] = mean
        
        # Compute variance
        var_sum = 0.0
        for c in range(start_c, end_c):
            diff = gConv[n, c, h, w] - mean
            var_sum += diff * diff
        gVar[n, group, h, w] = var_sum / group_size

@cute.jit
def compute_group_stats_host(
    mConv: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor,
    N: int, C_out: int, H_out: int, W_out: int, groups: int
):
    threads_x = 16
    threads_y = 16
    threads_z = 4
    
    blocks_x = cute.ceil_div(H_out * W_out, threads_x)
    blocks_y = cute.ceil_div(groups, threads_y)
    blocks_z = cute.ceil_div(N, threads_z)
    
    compute_group_stats_kernel(
        mConv, mMean, mVar, N, C_out, H_out, W_out, groups
    ).launch(
        grid=(blocks_x, blocks_y, blocks_z),
        block=(threads_x, threads_y, threads_z)
    )

@cute.kernel
def final_logsumexp_kernel(
    gLogSumExp: cute.Tensor, gFinal: cute.Tensor,
    N: int, H_out: int, W_out: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    hw = bidx * bdimx + tidx
    n = bidz * bdimz + tidz
    
    if n < N and hw < H_out * W_out:
        h = hw // W_out
        w = hw % W_out
        
        logsumexp_val = gLogSumExp[n, 0, h, w].log()
        gFinal[n, 0, h, w] = logsumexp_val

@cute.jit
def final_logsumexp_host(
    mLogSumExp: cute.Tensor, mFinal: cute.Tensor,
    N: int, H_out: int, W_out: int
):
    threads_x = 16
    threads_y = 1
    threads_z = 4
    
    blocks_x = cute.ceil_div(H_out * W_out, threads_x)
    blocks_y = 1
    blocks_z = cute.ceil_div(N, threads_z)
    
    final_logsumexp_kernel(
        mLogSumExp, mFinal, N, H_out, W_out
    ).launch(
        grid=(blocks_x, blocks_y, blocks_z),
        block=(threads_x, threads_y, threads_z)
    )

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.groups = groups
        self.eps = eps
        self.compiled = {}
        
    def forward(self, x):
        N, C_in, H_in, W_in = x.shape
        C_out = self.conv.out_channels
        H_out = H_in - self.conv.kernel_size[0] + 1
        W_out = W_in - self.conv.kernel_size[1] + 1
        
        x = x.contiguous().cuda()
        
        # Allocate intermediate tensors
        conv_out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        group_mean = torch.empty((N, self.groups, H_out, W_out), dtype=x.dtype, device=x.device)
        group_var = torch.empty((N, self.groups, H_out, W_out), dtype=x.dtype, device=x.device)
        logsumexp_temp = torch.zeros((N, 1, H_out, W_out), dtype=x.dtype, device=x.device)
        final_output = torch.empty((N, 1, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(self.conv.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(self.conv.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGamma = from_dlpack(self.group_norm.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(self.group_norm.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mGroupMean = from_dlpack(group_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mGroupVar = from_dlpack(group_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mLogSumExpTemp = from_dlpack(logsumexp_temp, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mFinalOutput = from_dlpack(final_output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        # Compile and run group stats computation
        key_stats = (x.dtype, "group_stats")
        compiled_stats = self.compiled.get(key_stats)
        if compiled_stats is None:
            compiled_stats = cute.compile(compute_group_stats_host, mConvOut, mGroupMean, mGroupVar, N, C_out, H_out, W_out, self.groups)
            self.compiled[key_stats] = compiled_stats
        
        # First compute convolution
        conv_out = torch.nn.functional.conv2d(x, self.conv.weight, self.conv.bias)
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        # Compute group statistics
        compiled_stats(mConvOut, mGroupMean, mGroupVar, N, C_out, H_out, W_out, self.groups)
        
        # Compile and run fused kernel
        key_fused = (x.dtype, "fused")
        compiled_fused = self.compiled.get(key_fused)
        if compiled_fused is None:
            compiled_fused = cute.compile(fused_conv_gn_tanh_hardswish_residual_logsumexp_host, 
                                        mX, mW, mB, mGroupMean, mGroupVar, mGamma, mBeta,
                                        mConvOut, mLogSumExpTemp, N, C_out, H_out, W_out, self.groups, self.eps)
            self.compiled[key_fused] = compiled_fused
        
        # Run fused kernel
        compiled_fused(mX, mW, mB, mGroupMean, mGroupVar, mGamma, mBeta,
                      mConvOut, mLogSumExpTemp, N, C_out, H_out, W_out, self.groups, self.eps)
        
        # Final logsumexp computation
        key_final = (x.dtype, "final_logsumexp")
        compiled_final = self.compiled.get(key_final)
        if compiled_final is None:
            compiled_final = cute.compile(final_logsumexp_host, mLogSumExpTemp, mFinalOutput, N, H_out, W_out)
            self.compiled[key_final] = compiled_final
        
        compiled_final(mLogSumExpTemp, mFinalOutput, N, H_out, W_out)
        
        return final_output