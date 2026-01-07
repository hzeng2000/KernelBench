import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_gemm_sigmoid_sum_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int,
    alpha: float, beta: float
):
    # Shared memory for tile of A and B
    smem_A = cute.shared_tensor((64, 32), dtype=cute.float32)
    smem_B = cute.shared_tensor((32, 64), dtype=cute.float32)
    
    # Thread-local accumulators
    accum = cute.local_tensor((1,), dtype=cute.float32, init=0.0)
    
    # Thread identifiers
    tid_x = cute.thread_idx_x()
    tid_y = cute.thread_idx_y()
    
    # Block identifiers
    bid_x = cute.block_idx_x()
    bid_y = cute.block_idx_y()
    
    # Compute global thread coordinates
    row = bid_y * 64 + tid_y
    col = bid_x * 64 + tid_x
    
    # Iterate over K dimension in tiles
    for tile_k in range(0, K, 32):
        # Load tile of A into shared memory
        if row < M and (tile_k + tid_x) < K:
            smem_A[tid_y, tid_x] = gA[row, tile_k + tid_x]
        else:
            smem_A[tid_y, tid_x] = 0.0
            
        # Load tile of B into shared memory
        if (tile_k + tid_y) < K and col < N:
            smem_B[tid_y, tid_x] = gB[tile_k + tid_y, col]
        else:
            smem_B[tid_y, tid_x] = 0.0
            
        cute.sync_threads()
        
        # Compute partial dot product
        for k in range(32):
            accum[0] += smem_A[tid_y, k] * smem_B[k, tid_x]
            
        cute.sync_threads()
    
    # Apply sigmoid and store result
    if row < M and col < N:
        val = alpha * accum[0] + beta
        sigmoid_val = 1.0 / (1.0 + cute.exp(-val))
        gC[row, col] = sigmoid_val

@cute.kernel
def reduce_sum_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    M: int, N: int
):
    # Shared memory for reduction
    smem = cute.shared_tensor((128,), dtype=cute.float32)
    
    # Thread identifiers
    tid = cute.thread_idx_x()
    bid = cute.block_idx_x()
    
    # Each thread reduces multiple elements
    row = bid
    if row < M:
        sum_val = 0.0
        # Grid-stride loop
        for col in range(tid, N, 128):
            sum_val += gInput[row, col]
        
        smem[tid] = sum_val
        cute.sync_threads()
        
        # Reduce within block
        for stride in [64, 32, 16, 8, 4, 2, 1]:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            cute.sync_threads()
            
        if tid == 0:
            gOutput[row, 0] = smem[0]

@cute.jit
def fused_gemm_sigmoid_sum_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    mSum: cute.Tensor
):
    M, K = mA.shape
    K_, N = mB.shape
    assert K == K_
    
    # Launch GEMM + Sigmoid kernel
    threads_per_block = (64, 64, 1)
    blocks_per_grid = (cute.ceil_div(N, 64), cute.ceil_div(M, 64), 1)
    
    fused_gemm_sigmoid_sum_kernel(
        mA, mB, mC,
        M, N, K,
        1.0, 0.0
    ).launch(grid=blocks_per_grid, block=threads_per_block)
    
    # Launch reduction kernel
    reduce_blocks = M
    reduce_threads = 128
    
    reduce_sum_kernel(
        mC, mSum,
        M, N
    ).launch(grid=(reduce_blocks, 1, 1), block=(reduce_threads, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size, bias=False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.compiled = {}
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Ensure contiguous memory layout
        x = x.contiguous().cuda()
        
        # Get weight tensor
        weight = self.linear.weight.contiguous().t().cuda()
        
        # Allocate output tensors
        intermediate = torch.empty((batch_size, self.hidden_size), dtype=torch.float32, device=x.device)
        output = torch.empty((batch_size, 1), dtype=torch.float32, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(intermediate, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mSum = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile and launch kernel
        key = (x.dtype, weight.dtype)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_sigmoid_sum_host, mA, mB, mC, mSum)
            self.compiled[key] = compiled
            
        compiled(mA, mB, mC, mSum)
        
        return output