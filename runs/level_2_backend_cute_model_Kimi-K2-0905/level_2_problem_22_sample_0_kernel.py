import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_matmul_scale_residual_clamp_lse_mish_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    scale_factor: float, clamp_min: float, clamp_max: float
):
    # Shared memory for tile
    shared_mem = cute.shared_memory(128 * 128 * 4 + 128 * 128 * 4)  # 2 tiles
    
    # Thread identifiers
    tidx = cute.thread_idx()
    tidy = cute.thread_idx_y()
    
    # Block identifiers
    bidx = cute.block_idx()
    bidy = cute.block_idx_y()
    
    # Tile sizes
    TILE_M = 128
    TILE_N = 128
    TILE_K = 32
    
    # Global indices
    row = bidx * TILE_M + tidx
    col = bidy * TILE_N + tidy
    
    # Matrix dimensions
    M = gX.shape[0]
    N = gW.shape[0]
    K = gX.shape[1]
    
    if row < M and col < N:
        # Compute matmul + scale + residual + clamp
        acc = 0.0
        for k in range(K):
            acc += gX[row, k] * gW[col, k]
        
        # Add bias
        acc += gB[col]
        
        # Scale and residual (x + x)
        acc = acc * scale_factor * 2.0
        
        # Clamp
        acc = cute.max(cute.min(acc, clamp_max), clamp_min)
        
        # Store intermediate result
        gOut[row, col] = acc

@cute.kernel
def logsumexp_mish_kernel(gIn: cute.Tensor, gOut: cute.Tensor):
    # Thread identifiers
    tidx = cute.thread_idx()
    bidx = cute.block_idx()
    
    # Block dimensions
    block_size = 256
    
    # Shared memory for reduction
    shared_mem = cute.shared_memory(block_size * 4)
    
    # Global indices
    row = bidx
    col = tidx
    
    # Dimensions
    M = gIn.shape[0]
    N = gIn.shape[1]
    
    if row < M:
        # Load data and compute max for numerical stability
        max_val = -1e20
        for i in range(col, N, block_size):
            val = gIn[row, i]
            max_val = cute.max(max_val, val)
        
        # Block reduction for max
        shared_mem[tidx] = max_val
        cute.sync_threads()
        
        # Reduce within block
        for offset in [128, 64, 32, 16, 8, 4, 2, 1]:
            if tidx < offset and tidx + offset < block_size:
                shared_mem[tidx] = cute.max(shared_mem[tidx], shared_mem[tidx + offset])
            cute.sync_threads()
        
        max_val = shared_mem[0]
        cute.sync_threads()
        
        # Compute exp(x - max) and sum
        sum_exp = 0.0
        for i in range(col, N, block_size):
            val = gIn[row, i]
            exp_val = cute.exp(val - max_val)
            sum_exp += exp_val
        
        # Block reduction for sum
        shared_mem[tidx] = sum_exp
        cute.sync_threads()
        
        for offset in [128, 64, 32, 16, 8, 4, 2, 1]:
            if tidx < offset and tidx + offset < block_size:
                shared_mem[tidx] += shared_mem[tidx + offset]
            cute.sync_threads()
        
        sum_exp = shared_mem[0]
        
        if tidx == 0:
            # Compute log(sum(exp(x - max))) + max
            lse = cute.log(sum_exp) + max_val
            
            # Compute mish activation
            tanh_arg = cute.log(1.0 + cute.exp(lse))
            tanh_val = cute.tanh(tanh_arg)
            mish = lse * tanh_val
            
            # Final output
            gOut[row, 0] = mish

@cute.jit
def fused_matmul_scale_residual_clamp_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    scale_factor: float, clamp_min: float, clamp_max: float
):
    M = mX.shape[0]
    N = mW.shape[0]
    
    threads_per_block = (128, 128)
    grid = (cute.ceil_div(M, 128), cute.ceil_div(N, 128))
    
    fused_matmul_scale_residual_clamp_lse_mish_kernel(
        mX, mW, mB, mOut, scale_factor, clamp_min, clamp_max
    ).launch(grid=grid, block=threads_per_block)

@cute.jit
def logsumexp_mish_host(mIn: cute.Tensor, mOut: cute.Tensor):
    M = mIn.shape[0]
    
    threads_per_block = 256
    grid = M
    
    logsumexp_mish_kernel(mIn, mOut).launch(grid=grid, block=threads_per_block)

class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Intermediate tensor for matmul result
        intermediate = torch.empty((batch_size, self.weight.shape[0]), dtype=x.dtype, device=x.device)
        
        # Final output tensor
        output = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mIntermediate = from_dlpack(intermediate, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mOut = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile and run fused matmul kernel
        key = (x.dtype,)
        compiled_matmul = self.compiled.get(key)
        if compiled_matmul is None:
            compiled_matmul = cute.compile(
                fused_matmul_scale_residual_clamp_host,
                mX, mW, mB, mIntermediate,
                self.scale_factor, self.clamp_min, self.clamp_max
            )
            self.compiled[key] = compiled_matmul
        
        compiled_matmul(mX, mW, mB, mIntermediate, self.scale_factor, self.clamp_min, self.clamp_max)
        
        # Compile and run logsumexp + mish kernel
        compiled_lse = self.compiled.get(key + ('lse',))
        if compiled_lse is None:
            compiled_lse = cute.compile(logsumexp_mish_host, mIntermediate, mOut)
            self.compiled[key + ('lse',)] = compiled_lse
        
        compiled_lse(mIntermediate, mOut)
        
        return output