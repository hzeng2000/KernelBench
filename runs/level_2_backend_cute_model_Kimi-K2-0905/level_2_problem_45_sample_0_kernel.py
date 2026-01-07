import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_sigmoid_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int
):
    # Shared memory for tile of A and B
    shared_A = cute.shared_memory(size=(64, 64), dtype=cute.float32)
    shared_B = cute.shared_memory(size=(64, 64), dtype=cute.float32)
    
    # Thread-local accumulator
    acc = cute.local_tensor(shape=(1,), dtype=cute.float32, init=0.0)
    
    # Tile indices
    tile_m = cute.program_id(0)
    tile_n = cute.program_id(1)
    
    # Thread indices within warp
    thread_m = cute.thread_idx() // 16
    thread_n = cute.thread_idx() % 16
    
    # Global row/col
    global_m = tile_m * 64 + thread_m * 4
    global_n = tile_n * 64 + thread_n * 4
    
    # Compute GEMM tile
    for k_tile in range(0, K, 64):
        # Load tile of A
        for i in range(4):
            if global_m + i < M and k_tile + thread_n < K:
                shared_A[thread_m * 4 + i, thread_n] = gA[global_m + i, k_tile + thread_n]
        
        # Load tile of B
        for i in range(4):
            if global_n + i < N and k_tile + thread_m < K:
                shared_B[thread_m, thread_n * 4 + i] = gB[k_tile + thread_m, global_n + i]
        
        cute.sync_threads()
        
        # Compute partial dot products
        for k in range(64):
            if k_tile + k < K:
                for i in range(4):
                    for j in range(4):
                        if global_m + i < M and global_n + j < N:
                            a_val = shared_A[thread_m * 4 + i, k]
                            b_val = shared_B[k, thread_n * 4 + j]
                            acc[0] += a_val * b_val
        
        cute.sync_threads()
    
    # Apply sigmoid and store result
    for i in range(4):
        for j in range(4):
            if global_m + i < M and global_n + j < N:
                val = acc[0]
                sigmoid_val = 1.0 / (1.0 + cute.exp(-val))
                gC[global_m + i, global_n + j] = sigmoid_val

@cute.kernel
def gemm_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int
):
    # Shared memory for tile of A and B
    shared_A = cute.shared_memory(size=(64, 64), dtype=cute.float32)
    shared_B = cute.shared_memory(size=(64, 64), dtype=cute.float32)
    
    # Thread-local accumulator
    acc = cute.local_tensor(shape=(1,), dtype=cute.float32, init=0.0)
    
    # Tile indices
    tile_m = cute.program_id(0)
    tile_n = cute.program_id(1)
    
    # Thread indices within warp
    thread_m = cute.thread_idx() // 16
    thread_n = cute.thread_idx() % 16
    
    # Global row/col
    global_m = tile_m * 64 + thread_m * 4
    global_n = tile_n * 64 + thread_n * 4
    
    # Compute GEMM tile
    for k_tile in range(0, K, 64):
        # Load tile of A
        for i in range(4):
            if global_m + i < M and k_tile + thread_n < K:
                shared_A[thread_m * 4 + i, thread_n] = gA[global_m + i, k_tile + thread_n]
        
        # Load tile of B
        for i in range(4):
            if global_n + i < N and k_tile + thread_m < K:
                shared_B[thread_m, thread_n * 4 + i] = gB[k_tile + thread_m, global_n + i]
        
        cute.sync_threads()
        
        # Compute partial dot products
        for k in range(64):
            if k_tile + k < K:
                for i in range(4):
                    for j in range(4):
                        if global_m + i < M and global_n + j < N:
                            a_val = shared_A[thread_m * 4 + i, k]
                            b_val = shared_B[k, thread_n * 4 + j]
                            acc[0] += a_val * b_val
        
        cute.sync_threads()
    
    # Store result
    for i in range(4):
        for j in range(4):
            if global_m + i < M and global_n + j < N:
                gC[global_m + i, global_n + j] = acc[0]

@cute.kernel
def logsumexp_kernel(gX: cute.Tensor, gY: cute.Tensor, M: int, N: int):
    # Shared memory for reduction
    shared_max = cute.shared_memory(size=(256,), dtype=cute.float32)
    shared_sum = cute.shared_memory(size=(256,), dtype=cute.float32)
    
    # Thread indices
    row = cute.block_idx() * cute.block_dim() + cute.thread_idx()
    
    if row < M:
        # Find max in row
        max_val = cute.float32(-1e38)
        for col in range(N):
            val = gX[row, col]
            if val > max_val:
                max_val = val
        
        # Compute sum of exp(x - max)
        sum_val = cute.float32(0.0)
        for col in range(N):
            val = gX[row, col]
            exp_val = cute.exp(val - max_val)
            sum_val += exp_val
        
        # Compute log(sum(exp(x - max))) + max
        logsumexp_val = cute.log(sum_val) + max_val
        gY[row] = logsumexp_val

@cute.jit
def gemm_sigmoid_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, M: int, N: int, K: int):
    grid_m = cute.ceil_div(M, 64)
    grid_n = cute.ceil_div(N, 64)
    gemm_sigmoid_kernel(mA, mB, mC, M, N, K).launch(grid=(grid_m, grid_n, 1), block=(256, 1, 1))

@cute.jit
def gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, M: int, N: int, K: int):
    grid_m = cute.ceil_div(M, 64)
    grid_n = cute.ceil_div(N, 64)
    gemm_kernel(mA, mB, mC, M, N, K).launch(grid=(grid_m, grid_n, 1), block=(256, 1, 1))

@cute.jit
def logsumexp_host(mX: cute.Tensor, mY: cute.Tensor, M: int, N: int):
    threads_per_block = 256
    grid_size = cute.ceil_div(M, threads_per_block)
    logsumexp_kernel(mX, mY, M, N).launch(grid=(grid_size, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.weight1 = nn.Parameter(torch.randn(hidden_size, input_size) * math.sqrt(2.0 / input_size))
        self.bias1 = nn.Parameter(torch.zeros(hidden_size))
        self.weight2 = nn.Parameter(torch.randn(output_size, hidden_size) * math.sqrt(2.0 / hidden_size))
        self.bias2 = nn.Parameter(torch.zeros(output_size))
        
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        
        # First linear + sigmoid fused
        x1 = torch.empty(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        mx = from_dlpack(x.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mw1 = from_dlpack(self.weight1.t().contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mx1 = from_dlpack(x1, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key1 = (x.dtype,)
        compiled1 = self.compiled.get(key1)
        if compiled1 is None:
            compiled1 = cute.compile(gemm_sigmoid_host, mx, mw1, mx1, batch_size, self.hidden_size, self.input_size)
            self.compiled[key1] = compiled1
        compiled1(mx, mw1, mx1, batch_size, self.hidden_size, self.input_size)
        
        # Add bias1
        x1 += self.bias1.unsqueeze(0)
        
        # Second linear
        x2 = torch.empty(batch_size, self.output_size, dtype=x.dtype, device=x.device)
        mx1 = from_dlpack(x1.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mw2 = from_dlpack(self.weight2.t().contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mx2 = from_dlpack(x2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key2 = (x.dtype,)
        compiled2 = self.compiled.get(key2)
        if compiled2 is None:
            compiled2 = cute.compile(gemm_host, mx1, mw2, mx2, batch_size, self.output_size, self.hidden_size)
            self.compiled[key2] = compiled2
        compiled2(mx1, mw2, mx2, batch_size, self.output_size, self.hidden_size)
        
        # Add bias2
        x2 += self.bias2.unsqueeze(0)
        
        # LogSumExp
        output = torch.empty(batch_size, dtype=x.dtype, device=x.device)
        mx2 = from_dlpack(x2.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mout = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key3 = (x.dtype,)
        compiled3 = self.compiled.get(key3)
        if compiled3 is None:
            compiled3 = cute.compile(logsumexp_host, mx2, mout, batch_size, self.output_size)
            self.compiled[key3] = compiled3
        compiled3(mx2, mout, batch_size, self.output_size)
        
        return output