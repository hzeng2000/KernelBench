import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_bias_relu_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, gBias: cute.Tensor,
    alpha: float, beta: float
):
    # Shared memory tiles
    sA = cute.shared_tensor((64, 32), dtype=cute.float32)
    sB = cute.shared_tensor((32, 64), dtype=cute.float32)
    
    # Thread-local fragments
    fragA = cute.local_tensor((8,), dtype=cute.float32)
    fragB = cute.local_tensor((8,), dtype=cute.float32)
    accum = cute.local_tensor((8, 8), dtype=cute.float32)
    
    # Thread identifiers
    tidx = cute.thread_idx()
    tidy = cute.thread_idx() // 32
    tidz = cute.thread_idx() % 32
    
    # Block identifiers
    bx = cute.block_idx()
    by = cute.block_idx()
    
    # Matrix dimensions
    M = gA.shape[0]
    N = gB.shape[1]
    K = gA.shape[1]
    
    # Tile indices
    tile_m = by * 64
    tile_n = bx * 64
    
    # Initialize accumulator
    for i in range(8):
        for j in range(8):
            accum[i, j] = 0.0
    
    # Main loop over K dimension
    for k_tile in range(0, K, 32):
        # Load tile from A
        for i in range(8):
            for j in range(4):
                global_m = tile_m + tidy * 8 + i
                global_k = k_tile + tidz * 4 + j
                if global_m < M and global_k < K:
                    sA[tidy * 8 + i, tidz * 4 + j] = gA[global_m, global_k]
                else:
                    sA[tidy * 8 + i, tidz * 4 + j] = 0.0
        
        # Load tile from B
        for i in range(4):
            for j in range(8):
                global_k = k_tile + tidy * 4 + i
                global_n = tile_n + tidz * 8 + j
                if global_k < K and global_n < N:
                    sB[tidy * 4 + i, tidz * 8 + j] = gB[global_k, global_n]
                else:
                    sB[tidy * 4 + i, tidz * 8 + j] = 0.0
        
        cute.sync_threads()
        
        # Compute partial dot product
        for k in range(32):
            # Load fragments
            for i in range(8):
                fragA[i] = sA[tidy * 8 + i, k]
            for j in range(8):
                fragB[j] = sB[k, tidz * 8 + j]
            
            # Accumulate
            for i in range(8):
                for j in range(8):
                    accum[i, j] += fragA[i] * fragB[j]
        
        cute.sync_threads()
    
    # Store results with bias and ReLU
    for i in range(8):
        for j in range(8):
            global_m = tile_m + tidy * 8 + i
            global_n = tile_n + tidz * 8 + j
            if global_m < M and global_n < N:
                val = alpha * accum[i, j]
                if beta != 0.0:
                    val += beta * gBias[global_n]
                gC[global_m, global_n] = cute.max(val, 0.0)

@cute.jit
def gemm_bias_relu_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
    alpha: float, beta: float
):
    M = mA.shape[0]
    N = mB.shape[1]
    
    grid_x = cute.ceil_div(N, 64)
    grid_y = cute.ceil_div(M, 64)
    
    gemm_bias_relu_kernel(mA, mB, mC, mBias, alpha, beta).launch(
        grid=(grid_x, grid_y, 1), 
        block=(256, 1, 1)
    )

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = self.weight.shape[0]
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        C = torch.empty((batch_size, out_features), dtype=x.dtype, device=x.device)
        
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight.t(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(gemm_bias_relu_host, mA, mB, mC, mBias, 1.0, 1.0)
            self.compiled[key] = compiled
            
        compiled(mA, mB, mC, mBias, 1.0, 1.0)
        return C