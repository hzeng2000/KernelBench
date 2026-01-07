import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_swish_gn_swish_mult_swish_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gBias: cute.Tensor,
    gScale: cute.Tensor, gBiasGN: cute.Tensor, gWeight: cute.Tensor,
    gOut: cute.Tensor,
    M: int, N: int, K: int, num_groups: int, channels_per_group: int
):
    # Shared memory for tile
    smem_a = cute.shared_tensor((64, 32), dtype=cute.float32)
    smem_b = cute.shared_tensor((32, 64), dtype=cute.float32)
    smem_c = cute.shared_tensor((64, 64), dtype=cute.float32)
    
    # Thread identifiers
    tidx = cute.thread_idx_x()
    tidy = cute.thread_idx_y()
    bidx = cute.block_idx_x()
    bidy = cute.block_idx_y()
    
    # Tile indices
    tile_m = bidx * 64
    tile_n = bidy * 64
    
    # Accumulator
    acc = cute.zeros((64,), dtype=cute.float32)
    
    # GEMM computation
    for k_tile in range(0, K, 32):
        # Load A tile
        if tile_m + tidy < M and k_tile + tidx < K:
            smem_a[tidy, tidx] = gA[tile_m + tidy, k_tile + tidx]
        else:
            smem_a[tidy, tidx] = 0.0
            
        # Load B tile
        if k_tile + tidy < K and tile_n + tidx < N:
            smem_b[tidy, tidx] = gB[k_tile + tidy, tile_n + tidx]
        else:
            smem_b[tidy, tidx] = 0.0
            
        cute.sync_threads()
        
        # Compute partial GEMM
        for k in range(32):
            if tile_m + tidy < M and tile_n + tidx < N:
                acc += smem_a[tidy, k] * smem_b[k, tidx]
                
        cute.sync_threads()
    
    # Add bias and write intermediate result
    if tile_m + tidy < M and tile_n + tidx < N:
        acc += gBias[tile_n + tidx]
        smem_c[tidy, tidx] = acc
        
    cute.sync_threads()
    
    # GroupNorm + Swish fusion
    if tile_m + tidy < M and tile_n + tidx < N:
        row = tile_m + tidy
        col = tile_n + tidx
        
        # Compute group index
        group_idx = col // channels_per_group
        
        # Compute mean for group
        sum_val = cute.zeros((1,), dtype=cute.float32)
        for c in range(channels_per_group):
            g_col = group_idx * channels_per_group + c
            if g_col < N:
                sum_val += smem_c[tidy, c] if c < 64 else gOut[row, g_col]
                
        mean = sum_val / channels_per_group
        
        # Compute variance
        var_sum = cute.zeros((1,), dtype=cute.float32)
        for c in range(channels_per_group):
            g_col = group_idx * channels_per_group + c
            if g_col < N:
                val = smem_c[tidy, c] if c < 64 else gOut[row, g_col]
                var_sum += (val - mean) * (val - mean)
                
        var = var_sum / channels_per_group
        rstd = cute.rsqrt(var + 1e-5)
        
        # Normalize, scale, bias
        normalized = (smem_c[tidy, tidx] - mean) * rstd
        gn_out = normalized * gScale[col] + gBiasGN[col]
        
        # First Swish
        sig1 = 1.0 / (1.0 + cute.exp(-gn_out))
        swish1 = gn_out * sig1
        
        # Multiply with weight
        mult_out = swish1 * gWeight[col]
        
        # Second Swish
        sig2 = 1.0 / (1.0 + cute.exp(-mult_out))
        final_out = mult_out * sig2
        
        gOut[row, col] = final_out

@cute.jit
def fused_gemm_swish_gn_swish_mult_swish_host(
    mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor,
    mScale: cute.Tensor, mBiasGN: cute.Tensor, mWeight: cute.Tensor,
    mOut: cute.Tensor, M: int, N: int, K: int, num_groups: int
):
    channels_per_group = N // num_groups
    
    grid_x = cute.ceil_div(M, 64)
    grid_y = cute.ceil_div(N, 64)
    
    gemm_swish_gn_swish_mult_swish_kernel(
        mA, mB, mBias, mScale, mBiasGN, mWeight, mOut,
        M, N, K, num_groups, channels_per_group
    ).launch(grid=(grid_x, grid_y, 1), block=(64, 64, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # GroupNorm parameters
        self.group_norm_weight = nn.Parameter(torch.ones(out_features))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_features))
        
        # Multiply weight
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        
        self.compiled = None
        
    def forward(self, x):
        batch_size = x.shape[0]
        M = batch_size
        N = self.out_features
        K = self.in_features
        
        # Ensure contiguous and on CUDA
        x = x.contiguous().cuda()
        out = torch.empty((M, N), dtype=torch.float32, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.weight.t().contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(self.bias.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mScale = from_dlpack(self.group_norm_weight.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBiasGN = from_dlpack(self.group_norm_bias.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mWeight = from_dlpack(self.multiply_weight.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile if not already done
        if self.compiled is None:
            self.compiled = cute.compile(
                fused_gemm_swish_gn_swish_mult_swish_host,
                mA, mB, mBias, mScale, mBiasGN, mWeight, mOut,
                M, N, K, self.num_groups
            )
        
        # Launch kernel
        self.compiled(mA, mB, mBias, mScale, mBiasGN, mWeight, mOut, M, N, K, self.num_groups)
        
        return out