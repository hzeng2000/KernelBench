import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_gemm_bias_hardtanh_mish_gn_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, gBias: cute.Tensor,
    gOut: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor, gWeight: cute.Tensor, gBiasGN: cute.Tensor,
    M: int, N: int, K: int, num_groups: int, eps: float
):
    # Shared memory for tile
    shared_mem = cute.shared_memory(128 * 128 * 4 + 128 * 4)  # For A tile, B tile, and partial sums
    
    # Thread identifiers
    tidx = cute.arch.thread_idx().x
    tidy = cute.arch.thread_idx().y
    bidx = cute.arch.block_idx().x
    bidy = cute.arch.block_idx().y
    
    # Tile sizes
    TM = 128
    TN = 128
    TK = 32
    
    # Global thread indices
    row = bidx * TM + tidy
    col = bidy * TN + tidx
    
    # Accumulator
    acc = 0.0
    
    # GEMM computation
    for k in range(0, K, TK):
        # Load tiles from A and B
        if row < M and k + tidx < K:
            a_val = gA[row, k + tidx]
        else:
            a_val = 0.0
            
        if col < N and k + tidy < K:
            b_val = gB[k + tidy, col]
        else:
            b_val = 0.0
            
        # Compute partial dot product
        acc += a_val * b_val
    
    # Add bias
    if row < M and col < N:
        acc += gBias[col]
    
    # Hardtanh activation
    acc = cute.max(-1.0, cute.min(1.0, acc))
    
    # Mish activation
    tanh_arg = cute.log1p(cute.exp(acc))
    mish_val = acc * cute.tanh(tanh_arg)
    
    # Store intermediate result for group norm
    if row < M and col < N:
        gC[row, col] = mish_val
    
    # GroupNorm computation
    group_size = N // num_groups
    group_idx = col // group_size
    
    # Compute mean per group
    if row < M:
        sum_val = 0.0
        count = 0
        for c in range(group_idx * group_size, min((group_idx + 1) * group_size, N)):
            if c < N:
                sum_val += gC[row, c]
                count += 1
        
        mean = sum_val / count
        gMean[row, group_idx] = mean
        
        # Compute variance
        var_sum = 0.0
        for c in range(group_idx * group_size, min((group_idx + 1) * group_size, N)):
            if c < N:
                diff = gC[row, c] - mean
                var_sum += diff * diff
        
        var = var_sum / count
        gVar[row, group_idx] = var
        
        # Normalize
        for c in range(group_idx * group_size, min((group_idx + 1) * group_size, N)):
            if c < N and row < M:
                normalized = (gC[row, c] - mean) / cute.sqrt(var + eps)
                gOut[row, c] = normalized * gWeight[c] + gBiasGN[c]

@cute.jit
def fused_gemm_bias_hardtanh_mish_gn_host(
    mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor,
    mOut: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor, mWeight: cute.Tensor, mBiasGN: cute.Tensor,
    M: int, N: int, K: int, num_groups: int, eps: float
):
    # Intermediate tensor
    mC = cute.empty((M, N), dtype=mA.dtype)
    
    # Grid dimensions
    grid_x = cute.ceil_div(M, 128)
    grid_y = cute.ceil_div(N, 128)
    
    fused_gemm_bias_hardtanh_mish_gn_kernel(
        mA, mB, mC, mBias, mOut, mMean, mVar, mWeight, mBiasGN,
        M, N, K, num_groups, eps
    ).launch(grid=(grid_x, grid_y, 1), block=(128, 128, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Initialize weight and bias for linear layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # GroupNorm parameters
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        M = batch_size
        N = self.out_features
        K = self.in_features
        
        # Ensure contiguous and on CUDA
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        # Output tensor
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        
        # GroupNorm statistics tensors
        mean = torch.empty((M, self.num_groups), dtype=x.dtype, device=x.device)
        var = torch.empty((M, self.num_groups), dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight.T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Get GroupNorm weight and bias
        gn_weight = self.group_norm.weight.contiguous().cuda()
        gn_bias = self.group_norm.bias.contiguous().cuda()
        mWeight = from_dlpack(gn_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBiasGN = from_dlpack(gn_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        # Compile and launch kernel
        key = (x.dtype, self.num_groups)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_gemm_bias_hardtanh_mish_gn_host,
                mA, mB, mBias, mOut, mMean, mVar, mWeight, mBiasGN,
                M, N, K, self.num_groups, self.group_norm.eps
            )
            self.compiled[key] = compiled
        
        compiled(mA, mB, mBias, mOut, mMean, mVar, mWeight, mBiasGN, M, N, K, self.num_groups, self.group_norm.eps)
        
        return out