import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_scale_hardtanh_gelu_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    scale: float, min_val: float, max_val: float
):
    # Shared memory for tile
    shared_mem = cute.shared_memory(128 * 128 * 4 + 128 * 128 * 4)  # A tile + B tile
    
    # Tile dimensions
    TILE_M = 128
    TILE_N = 128
    TILE_K = 32
    
    # Thread identifiers
    tidx = cute.thread_idx()
    tidy = cute.thread_idx_y()
    
    # Global thread indices
    thread_m = tidx % 16
    thread_n = tidx // 16
    
    # Block indices
    block_m = cute.block_idx()
    block_n = cute.block_idx_y()
    
    # Global memory indices
    m = block_m * TILE_M + thread_m * 8
    n = block_n * TILE_N + thread_n * 8
    
    # Accumulator registers
    acc = cute.zeros(8, 8, dtype=cute.float32)
    
    # Main loop over K dimension
    K = gA.shape[1]
    for k in range(0, K, TILE_K):
        # Load A tile
        for i in range(8):
            for j in range(8):
                if m + i < gA.shape[0] and k + tidy < gA.shape[1]:
                    shared_mem[tidy * TILE_M + thread_m * 8 + i] = gA[m + i, k + tidy]
        
        # Load B tile
        for i in range(8):
            for j in range(8):
                if k + tidy < gB.shape[0] and n + j < gB.shape[1]:
                    shared_mem[TILE_M * TILE_K + tidy * TILE_N + thread_n * 8 + j] = gB[k + tidy, n + j]
        
        cute.sync_threads()
        
        # Compute partial GEMM
        for kk in range(TILE_K):
            for i in range(8):
                for j in range(8):
                    a_val = shared_mem[kk * TILE_M + thread_m * 8 + i]
                    b_val = shared_mem[TILE_M * TILE_K + kk * TILE_N + thread_n * 8 + j]
                    acc[i, j] += a_val * b_val
        
        cute.sync_threads()
    
    # Apply scaling, hardtanh, and GELU
    for i in range(8):
        for j in range(8):
            if m + i < gC.shape[0] and n + j < gC.shape[1]:
                # Scale
                val = acc[i, j] * scale
                
                # Hardtanh
                val = cute.max(val, min_val)
                val = cute.min(val, max_val)
                
                # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x_cubed = val * val * val
                tanh_arg = 0.7978845608 * (val + 0.044715 * x_cubed)
                
                # Fast tanh approximation
                tanh_val = tanh_arg / (1.0 + cute.abs(tanh_arg))
                
                gelu_val = 0.5 * val * (1.0 + tanh_val)
                
                gC[m + i, n + j] = gelu_val

@cute.jit
def gemm_scale_hardtanh_gelu_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    scale: float, min_val: float, max_val: float
):
    M = mA.shape[0]
    N = mB.shape[1]
    
    block_dim = (256, 1, 1)
    grid_dim = (cute.ceil_div(M, 128), cute.ceil_div(N, 128), 1)
    
    gemm_scale_hardtanh_gelu_kernel(mA, mB, mC, scale, min_val, max_val).launch(
        grid=grid_dim, block=block_dim
    )

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialize weight using Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.compiled = None
        
    def forward(self, x):
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = self.weight.shape[0]
        
        # Ensure contiguous memory layout
        x = x.contiguous().cuda()
        weight_t = self.weight.t().contiguous().cuda()
        
        # Output tensor
        output = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight_t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile kernel if not already compiled
        if self.compiled is None:
            self.compiled = cute.compile(
                gemm_scale_hardtanh_gelu_host,
                mA, mB, mC,
                self.scaling_factor, self.hardtanh_min, self.hardtanh_max
            )
        
        # Launch kernel
        self.compiled(mA, mB, mC, self.scaling_factor, self.hardtanh_min, self.hardtanh_max)
        
        # Add bias
        output += self.bias
        
        return output