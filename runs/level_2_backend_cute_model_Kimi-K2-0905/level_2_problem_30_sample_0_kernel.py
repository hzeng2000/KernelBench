import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_groupnorm_hardtanh_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    M: int, N: int, K: int, num_groups: int, eps: float, min_val: float, max_val: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy
    
    if row < M and col < N:
        # GEMM computation
        sum_val = 0.0
        for k in range(K):
            sum_val += gX[row, k] * gW[col, k]
        sum_val += gB[col]
        gOut[row, col] = sum_val
        
        # Group normalization
        group_size = N // num_groups
        group_idx = col // group_size
        
        # Compute group mean
        if col == group_idx * group_size:
            group_sum = 0.0
            for i in range(group_size):
                if group_idx * group_size + i < N:
                    group_sum += gOut[row, group_idx * group_size + i]
            gMean[row, group_idx] = group_sum / group_size
        
        # Compute group variance
        if col == group_idx * group_size:
            group_var_sum = 0.0
            for i in range(group_size):
                if group_idx * group_size + i < N:
                    diff = gOut[row, group_idx * group_size + i] - gMean[row, group_idx]
                    group_var_sum += diff * diff
            gVar[row, group_idx] = group_var_sum / group_size
        
        # Normalize and apply scale/shift
        mean = gMean[row, group_idx]
        var = gVar[row, group_idx]
        normalized = (gOut[row, col] - mean) / cute.sqrt(var + eps)
        gOut[row, col] = normalized * gGamma[col] + gBeta[col]
        
        # HardTanh
        if gOut[row, col] < min_val:
            gOut[row, col] = min_val
        elif gOut[row, col] > max_val:
            gOut[row, col] = max_val

@cute.jit
def gemm_groupnorm_hardtanh_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    M: int, N: int, K: int, num_groups: int, eps: float, min_val: float, max_val: float
):
    threads_per_block = 16
    grid_x = cute.ceil_div(M, threads_per_block)
    grid_y = cute.ceil_div(N, threads_per_block)
    
    gemm_groupnorm_hardtanh_kernel(
        mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta,
        M, N, K, num_groups, eps, min_val, max_val
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block, threads_per_block, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.eps = 1e-5
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        out = torch.empty((batch_size, self.out_features), dtype=x.dtype, device=x.device)
        mean = torch.empty((batch_size, self.num_groups), dtype=x.dtype, device=x.device)
        var = torch.empty((batch_size, self.num_groups), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGamma = from_dlpack(self.gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(self.beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                gemm_groupnorm_hardtanh_host,
                mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta,
                batch_size, self.out_features, self.in_features,
                self.num_groups, self.eps, self.hardtanh_min, self.hardtanh_max
            )
            self.compiled[key] = compiled
        
        compiled(
            mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta,
            batch_size, self.out_features, self.in_features,
            self.num_groups, self.eps, self.hardtanh_min, self.hardtanh_max
        )
        return out