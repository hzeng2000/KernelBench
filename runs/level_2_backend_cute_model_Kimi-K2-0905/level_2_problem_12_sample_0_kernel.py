import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_mul_leakyrelu_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int,
    multiplier: float, negative_slope: float
):
    # Shared memory tile sizes
    BM = 128
    BN = 128
    BK = 8
    
    # Thread block and thread indices
    bx = cute.arch.block_idx().x
    by = cute.arch.block_idx().y
    tx = cute.arch.thread_idx().x
    ty = cute.arch.thread_idx().y
    
    # Compute global thread index
    thread_idx = ty * cute.arch.block_dim().x + tx
    
    # Shared memory for tiles
    sA = cute.shared_tensor((BM, BK), dtype=cute.float32)
    sB = cute.shared_tensor((BK, BN), dtype=cute.float32)
    
    # Compute row and column for this thread
    row = by * BM + ty
    col = bx * BN + tx
    
    # Accumulator
    acc = 0.0
    
    # Loop over K dimension
    for k in range(0, K, BK):
        # Load tile from A
        if row < M and (k + tx) < K:
            sA[ty, tx] = gA[row, k + tx]
        else:
            sA[ty, tx] = 0.0
            
        # Load tile from B
        if (k + ty) < K and col < N:
            sB[ty, tx] = gB[k + ty, col]
        else:
            sB[ty, tx] = 0.0
            
        cute.arch.sync_threads()
        
        # Compute partial dot product
        for i in range(BK):
            acc += sA[ty, i] * sB[i, tx]
            
        cute.arch.sync_threads()
    
    # Write result with multiplier and LeakyReLU
    if row < M and col < N:
        val = acc * multiplier
        if val < 0:
            val = val * negative_slope
        gC[row, col] = val

@cute.jit
def gemm_mul_leakyrelu_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    M: int, N: int, K: int,
    multiplier: float, negative_slope: float
):
    threads_per_block = 16
    grid_x = cute.ceil_div(N, 128)
    grid_y = cute.ceil_div(M, 128)
    
    gemm_mul_leakyrelu_kernel(
        mA, mB, mC, M, N, K, multiplier, negative_slope
    ).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, threads_per_block, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.compiled = {}
        
    def forward(self, x):
        M = x.shape[0]
        K = x.shape[1]
        N = self.weight.shape[0]
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        output = torch.empty((M, N), dtype=x.dtype, device=x.device)
        
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight.t(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Add bias separately for simplicity
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                gemm_mul_leakyrelu_host,
                mA, mB, mC, M, N, K, self.multiplier, self.negative_slope
            )
            self.compiled[key] = compiled
            
        # Recompute with fused kernel
        compiled(mA, mB, mC, M, N, K, self.multiplier, self.negative_slope)
        
        # Add bias manually
        output += self.bias
        
        return output