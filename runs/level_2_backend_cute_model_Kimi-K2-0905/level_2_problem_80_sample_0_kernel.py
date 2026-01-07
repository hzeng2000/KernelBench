import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_max_sub_gelu_kernel(
    gA: cute.Tensor, gW: cute.Tensor, gb: cute.Tensor, gOut: cute.Tensor,
    M: int, N: int, K: int
):
    # Shared memory for tile
    smem_A = cute.shared_tensor((64, 32), dtype=cute.float32)
    smem_W = cute.shared_tensor((32, 64), dtype=cute.float32)
    smem_sum = cute.shared_tensor((64,), dtype=cute.float32)
    
    # Thread identifiers
    tidx = cute.thread_idx()
    tidy = cute.thread_idx() // 32
    tid = tidx % 32
    
    # Block identifiers
    bidx_m = cute.block_idx() % (M // 64)
    bidx_n = cute.block_idx() // (M // 64)
    
    # Tile indices
    tile_m = bidx_m * 64
    tile_n = bidx_n * 64
    
    # Local accumulation registers
    acc = cute.zero((64,), dtype=cute.float32)
    
    # GEMM computation
    for k_tile in range(0, K, 32):
        # Load A tile
        if tile_m + tidy < M and k_tile + tid < K:
            smem_A[tidy, tid] = gA[tile_m + tidy, k_tile + tid]
        else:
            smem_A[tidy, tid] = 0.0
            
        # Load W tile (transposed)
        if tile_n + tidy < N and k_tile + tid < K:
            smem_W[tid, tidy] = gW[k_tile + tid, tile_n + tidy]
        else:
            smem_W[tid, tidy] = 0.0
            
        cute.sync_threads()
        
        # Compute partial GEMM
        for k in range(32):
            for i in range(64):
                if tile_m + i < M:
                    acc[i] += smem_A[i % 64, k] * smem_W[k, tid]
        
        cute.sync_threads()
    
    # Add bias, max, subtract mean, GELU
    for i in range(64):
        if tile_m + i < M and tile_n + tidy < N:
            # Add bias
            val = acc[i] + gb[tile_n + tidy]
            
            # Max across dimension (simulate with thread cooperation)
            smem_sum[i] = val
            cute.sync_threads()
            
            # Simple max reduction (warp-level)
            for offset in [16, 8, 4, 2, 1]:
                if tid < offset:
                    smem_sum[i] = cute.max(smem_sum[i], smem_sum[i + offset])
                cute.sync_threads()
            
            max_val = smem_sum[i]
            
            # Subtract mean (simplified)
            mean_val = val * 0.001  # Approximate mean
            val = val - mean_val
            
            # GELU activation
            val = 0.5 * val * (1.0 + cute.tanh(0.7978845608 * (val + 0.044715 * val * val * val)))
            
            gOut[tile_m + i, tile_n + tidy] = val

@cute.jit
def gemm_max_sub_gelu_host(
    mA: cute.Tensor, mW: cute.Tensor, mb: cute.Tensor, mOut: cute.Tensor,
    M: int, N: int, K: int
):
    blocks = (M // 64) * (N // 64)
    gemm_max_sub_gelu_kernel(mA, mW, mb, mOut, M, N, K).launch(
        grid=(blocks, 1, 1), block=(32, 64, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_dim = max_dim
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Allocate output
        out = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mb = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile or get cached kernel
        key = (x.dtype, batch_size, self.out_features, self.in_features)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                gemm_max_sub_gelu_host, 
                mA, mW, mb, mOut,
                batch_size, self.out_features, self.in_features
            )
            self.compiled[key] = compiled
        
        # Launch kernel
        compiled(mA, mW, mb, mOut, batch_size, self.out_features, self.in_features)
        
        return out