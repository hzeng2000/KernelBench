import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_groupnorm_min_bias_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    gBias: cute.Tensor, M: int, N: int, K: int, num_groups: int, eps: float
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
        
        # GroupNorm computation
        group_size = N // num_groups
        group_idx = col // group_size
        
        # Compute group mean
        if col == group_idx * group_size:
            group_sum = 0.0
            for i in range(group_size):
                if col + i < N:
                    group_sum += sum_val
            gMean[row, group_idx] = group_sum / group_size
        
        # Compute group variance
        if col == group_idx * group_size:
            group_var_sum = 0.0
            mean = gMean[row, group_idx]
            for i in range(group_size):
                if col + i < N:
                    diff = sum_val - mean
                    group_var_sum += diff * diff
            gVar[row, group_idx] = group_var_sum / group_size
        
        # Normalize
        mean = gMean[row, group_idx]
        var = gVar[row, group_idx]
        normalized = (sum_val - mean) * cute.rsqrt(var + eps)
        
        # Scale and shift
        gamma = gGamma[col]
        beta = gBeta[col]
        norm_out = gamma * normalized + beta
        
        # Min reduction (simplified - store intermediate)
        if col == 0:
            min_val = norm_out
            for i in range(1, N):
                if row < M:
                    min_val = cute.min(min_val, norm_out)
            gOut[row, 0] = min_val + gBias[0, col, 0, 0]
        else:
            gOut[row, col] = norm_out + gBias[0, col, 0, 0]

@cute.jit
def gemm_groupnorm_min_bias_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    mBias: cute.Tensor, M: int, N: int, K: int, num_groups: int, eps: float
):
    threads_per_block = 256
    grid_x = cute.ceil_div(M, 16)
    grid_y = cute.ceil_div(N, 16)
    
    gemm_groupnorm_min_bias_kernel(
        mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta, mBias,
        M, N, K, num_groups, eps
    ).launch(grid=(grid_x, grid_y, 1), block=(16, 16, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_gemm = nn.Parameter(torch.randn(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.register_buffer('mean_buffer', torch.zeros(1, num_groups))
        self.register_buffer('var_buffer', torch.zeros(1, num_groups))
        
        self.compiled = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M, K = x.shape
        N = self.out_features
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias_gemm = self.bias_gemm.contiguous().cuda()
        gamma = self.gamma.contiguous().cuda()
        beta = self.beta.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        mean_buffer = torch.empty((M, self.num_groups), dtype=x.dtype, device=x.device)
        var_buffer = torch.empty((M, self.num_groups), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(bias_gemm, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mMean = from_dlpack(mean_buffer, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mVar = from_dlpack(var_buffer, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGamma = from_dlpack(gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype, M, N, K, self.num_groups)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                gemm_groupnorm_min_bias_host,
                mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta, mBias,
                M, N, K, self.num_groups, 1e-5
            )
            self.compiled[key] = compiled
        
        compiled(mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta, mBias, M, N, K, self.num_groups, 1e-5)
        
        # Apply min operation and bias
        min_vals = torch.min(out, dim=1, keepdim=True)[0]
        result = min_vals.unsqueeze(-1).unsqueeze(-1) + bias
        
        return result.squeeze()