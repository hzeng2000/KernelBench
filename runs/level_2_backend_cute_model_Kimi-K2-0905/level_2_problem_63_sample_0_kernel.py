import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_relu_div_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int, divisor: float
):
    # Shared memory for tile
    smem_A = cute.shared_tensor((64, 32), dtype=cute.float32)
    smem_B = cute.shared_tensor((32, 64), dtype=cute.float32)
    smem_C = cute.shared_tensor((64, 64), dtype=cute.float32)
    
    # Thread identifiers
    tidx = cute.thread_idx_x()
    tidy = cute.thread_idx_y()
    bidx = cute.block_idx_x()
    bidy = cute.block_idx_y()
    
    # Tile indices
    tile_m = bidx * 64
    tile_n = bidy * 64
    
    # Accumulator
    acc = cute.zeros((1,), dtype=cute.float32)
    
    # Loop over K dimension
    for k_tile in range(0, K, 32):
        # Load A tile
        if tile_m + tidy < M and k_tile + tidx < K:
            smem_A[tidy, tidx] = gA[tile_m + tidy, k_tile + tidx]
        else:
            smem_A[tidy, tidx] = 0.0
            
        # Load B tile
        if k_tile + tidy < K and tile_n + tidx < N:
            smem_B[tidy, tidx] = gB[k_tile + tidy, tile_n + tidx]
        else:
            smem_B[tidy, tidx] = 0.0
            
        cute.sync_threads()
        
        # Compute partial GEMM
        for k in range(32):
            if tile_m + tidy < M and tile_n + tidx < N:
                acc += smem_A[tidy, k] * smem_B[k, tidx]
                
        cute.sync_threads()
    
    # Apply ReLU and division
    if tile_m + tidy < M and tile_n + tidx < N:
        val = acc
        val = cute.max(val, 0.0)  # ReLU
        val = val / divisor        # Division
        gC[tile_m + tidy, tile_n + tidx] = val

@cute.jit
def fused_gemm_relu_div_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    M: int, N: int, K: int, divisor: float
):
    block_dim = (64, 64, 1)
    grid_dim = (
        cute.ceil_div(M, 64),
        cute.ceil_div(N, 64),
        1
    )
    
    fused_gemm_relu_div_kernel(
        mA, mB, mC, M, N, K, divisor
    ).launch(grid=grid_dim, block=block_dim)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.compiled = {}

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Transpose weight for GEMM
        weight_t = self.weight.t().contiguous()
        
        # Allocate output
        output = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight_t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile kernel if needed
        key = (x.dtype, self.divisor)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_gemm_relu_div_host,
                mA, mB, mC,
                batch_size, self.out_features, self.in_features,
                self.divisor
            )
            self.compiled[key] = compiled
        
        # Launch kernel
        compiled(mA, mB, mC, batch_size, self.out_features, self.in_features, self.divisor)
        
        # Add bias
        output += self.bias.unsqueeze(0)
        
        return output

import math