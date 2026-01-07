import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_matmul_dropout_softmax_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    dropout_p: float, seed: int
):
    # Shared memory for tile
    shared_mem = cute.shared_tensor((64, 64), dtype=cute.float32)
    
    # Thread indices
    tidx, tidy, _ = cute.arch.thread_idx()  
    bidx, bidy, _ = cute.arch.block_idx()  
    bdimx, bdimy, _ = cute.arch.block_dim()  

    # Matrix dimensions
    M = gX.shape[0]  # batch_size
    K = gX.shape[1]  # in_features
    N = gW.shape[0]  # out_features

    # Tile indices
    tile_m = bidx * 64 + tidx
    tile_n = bidy * 64 + tidy

    # Accumulator for matmul result
    acc = cute.float32(0.0)

    # Perform matmul for this tile
    for k in range(0, K, 4):
        # Load 4 elements from X and W
        if tile_m < M and k + 3 < K:
            x_vals = [
                gX[tile_m, k],
                gX[tile_m, k + 1] if k + 1 < K else cute.float32(0.0),
                gX[tile_m, k + 2] if k + 2 < K else cute.float32(0.0),
                gX[tile_m, k + 3] if k + 3 < K else cute.float32(0.0)
            ]
            
            for i in range(4):
                if tile_n < N and k + i < K:
                    w_val = gW[tile_n, k + i]
                    acc += x_vals[i] * w_val

    # Add bias
    if tile_m < M and tile_n < N:
        acc += gB[tile_n]

    # Apply dropout
    if tile_m < M and tile_n < N:
        # Simple LCG for random number generation
        rand_state = seed + tile_m * N + tile_n
        rand_state = (rand_state * 1103515245 + 12345) & 0x7fffffff
        rand_val = cute.float32(rand_state) / cute.float32(0x7fffffff)
        
        if rand_val < dropout_p:
            acc = cute.float32(0.0)
        else:
            acc = acc / (cute.float32(1.0) - dropout_p)

    # Store in shared memory for softmax
    if tidx < 64 and tidy < 64:
        shared_mem[tidx, tidy] = acc
    cute.arch.sync_threads()

    # Compute max for numerical stability (within tile)
    local_max = cute.float32(-1e20)
    for n in range(64):
        if bidy * 64 + n < N and tidx < 64:
            val = shared_mem[tidx, n]
            local_max = cute.max(local_max, val)
    
    # Compute exp and sum
    local_sum = cute.float32(0.0)
    for n in range(64):
        if bidy * 64 + n < N and tidx < 64:
            val = shared_mem[tidx, n]
            exp_val = cute.exp(val - local_max)
            shared_mem[tidx, n] = exp_val
            local_sum += exp_val
    
    # Normalize
    for n in range(64):
        if bidy * 64 + n < N and tidx < 64:
            val = shared_mem[tidx, n]
            if local_sum > cute.float32(0.0):
                gOut[tile_m, bidy * 64 + n] = val / local_sum
            else:
                gOut[tile_m, bidy * 64 + n] = cute.float32(0.0)

@cute.jit
def fused_matmul_dropout_softmax_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    dropout_p: float, seed: int
):
    M = mX.shape[0]
    N = mW.shape[0]

    threads_per_block = 256
    grid_x = cute.ceil_div(M, 64)
    grid_y = cute.ceil_div(N, 64)

    fused_matmul_dropout_softmax_kernel(
        mX, mW, mB, mOut, dropout_p, seed
    ).launch(grid=(grid_x, grid_y, 1), block=(16, 16, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.compiled = {}
        self.seed = 42

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        out = torch.empty((batch_size, self.out_features), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_matmul_dropout_softmax_host, mX, mW, mB, mOut, self.dropout_p, self.seed)
            self.compiled[key] = compiled

        self.seed += 1
        compiled(mX, mW, mB, mOut, self.dropout_p, self.seed)
        
        return out