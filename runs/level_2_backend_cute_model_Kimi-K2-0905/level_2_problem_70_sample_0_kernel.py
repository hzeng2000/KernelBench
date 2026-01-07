import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_sigmoid_scaling_residual_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    gD: cute.Tensor, scaling_factor: float, M: int, N: int, K: int
):
    # Shared memory for tile
    smem_A = cute.shared_tensor((64, 64), dtype=cute.float32)
    smem_B = cute.shared_tensor((64, 64), dtype=cute.float32)
    smem_C = cute.shared_tensor((64, 64), dtype=cute.float32)
    
    # Thread identifiers
    tidx = cute.thread_idx()
    tidy = cute.thread_idx()
    bidx = cute.block_idx()
    bidy = cute.block_idx()
    
    # Tile indices
    tile_m = bidx * 64
    tile_n = bidy * 64
    
    # Accumulator
    acc = cute.zeros((64, 64), dtype=cute.float32)
    
    # Main loop over K dimension
    for k_tile in range(0, K, 64):
        # Load tile from A
        for i in range(64):
            for j in range(64):
                if tile_m + i < M and k_tile + j < K:
                    smem_A[i, j] = gA[tile_m + i, k_tile + j]
                else:
                    smem_A[i, j] = 0.0
        
        # Load tile from B (transposed)
        for i in range(64):
            for j in range(64):
                if k_tile + i < K and tile_n + j < N:
                    smem_B[i, j] = gB[k_tile + i, tile_n + j]
                else:
                    smem_B[i, j] = 0.0
        
        # Synchronize
        cute.sync_threads()
        
        # Compute GEMM for this tile
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    acc[i, j] += smem_A[i, k] * smem_B[k, j]
        
        # Synchronize
        cute.sync_threads()
    
    # Apply sigmoid, scaling, and residual add
    for i in range(64):
        for j in range(64):
            if tile_m + i < M and tile_n + j < N:
                gemm_val = acc[i, j]
                sigmoid_val = 1.0 / (1.0 + cute.exp(-gemm_val))
                scaled_val = sigmoid_val * scaling_factor
                residual_val = scaled_val + gemm_val
                gD[tile_m + i, tile_n + j] = residual_val

@cute.jit
def gemm_sigmoid_scaling_residual_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mD: cute.Tensor,
    scaling_factor: float, M: int, N: int, K: int
):
    grid_m = cute.ceil_div(M, 64)
    grid_n = cute.ceil_div(N, 64)
    
    gemm_sigmoid_scaling_residual_kernel(
        mA, mB, mC, mD, scaling_factor, M, N, K
    ).launch(grid=(grid_m, grid_n, 1), block=(256, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.scaling_factor = scaling_factor
        self.compiled = {}
        
    def forward(self, x):
        batch_size = x.shape[0]
        input_size = x.shape[1]
        hidden_size = self.weight.shape[0]
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        output = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=x.device)
        
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(bias.unsqueeze(0).expand(batch_size, -1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mD = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key = (x.dtype, weight.dtype)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                gemm_sigmoid_scaling_residual_host,
                mA, mB, mC, mD,
                self.scaling_factor, batch_size, hidden_size, input_size
            )
            self.compiled[key] = compiled
        
        compiled(mA, mB, mC, mD, self.scaling_factor, batch_size, hidden_size, input_size)
        return output