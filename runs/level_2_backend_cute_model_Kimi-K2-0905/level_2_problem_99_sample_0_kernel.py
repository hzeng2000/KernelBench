import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_gelu_softmax_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    alpha: float, beta: float, M: int, N: int, K: int
):
    # Shared memory for tile
    shared_mem = cute.shared_memory(128 * 128 * 4 + 128 * 128 * 4)  # A tile + B tile
    
    # Thread identifiers
    tidx = cute.thread_idx_x()
    tidy = cute.thread_idx_y()
    bidx = cute.block_idx_x()
    bidy = cute.block_idx_y()
    
    # Tile sizes
    TILE_M = 128
    TILE_N = 128
    TILE_K = 32
    
    # Global memory access
    row = bidy * TILE_M + tidy
    col = bidx * TILE_N + tidx
    
    # Accumulator
    acc = 0.0
    
    # Loop over K dimension
    for k in range(0, K, TILE_K):
        # Load A tile
        if row < M and k + tidx < K:
            shared_mem[tidy * TILE_K + tidx] = gA[row * K + k + tidx]
        else:
            shared_mem[tidy * TILE_K + tidx] = 0.0
            
        # Load B tile
        if col < N and k + tidy < K:
            shared_mem[TILE_M * TILE_K + tidy * TILE_N + tidx] = gB[(k + tidy) * N + col]
        else:
            shared_mem[TILE_M * TILE_K + tidy * TILE_N + tidx] = 0.0
            
        cute.sync_threads()
        
        # Compute partial dot product
        for ki in range(TILE_K):
            a_val = shared_mem[tidy * TILE_K + ki]
            b_val = shared_mem[TILE_M * TILE_K + ki * TILE_N + tidx]
            acc += a_val * b_val
            
        cute.sync_threads()
    
    # Apply bias and GELU
    if row < M and col < N:
        # Add bias (assuming gC is bias)
        bias_val = gC[col] if beta != 0.0 else 0.0
        result = alpha * acc + beta * bias_val
        
        # GELU activation
        gelu_const = 0.7978845608  # sqrt(2/pi)
        gelu_scale = 0.044715
        tanh_arg = gelu_const * (result + gelu_scale * result * result * result)
        
        # Approximate tanh with sigmoid
        sigmoid_arg = 2.0 * tanh_arg
        tanh_approx = (2.0 / (1.0 + cute.exp(-sigmoid_arg))) - 1.0
        
        gelu_result = 0.5 * result * (1.0 + tanh_approx)
        
        # Store in global memory for softmax
        gA[row * N + col] = gelu_result  # Reuse A as temporary storage

@cute.kernel
def softmax_kernel(gX: cute.Tensor, gY: cute.Tensor, M: int, N: int):
    # Thread identifiers
    tidx = cute.thread_idx_x()
    bidx = cute.block_idx_x()
    
    # Each block handles one row
    row = bidx
    if row >= M:
        return
    
    # Shared memory for max and sum reduction
    shared_max = cute.shared_memory(256 * 4)
    shared_sum = cute.shared_memory(256 * 4 + 256 * 4)
    
    # Find max value in row
    max_val = -float('inf')
    for col in range(tidx, N, 256):
        val = gX[row * N + col]
        if val > max_val:
            max_val = val
    
    shared_max[tidx] = max_val
    cute.sync_threads()
    
    # Reduction for max
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        if tidx < stride and tidx + stride < 256:
            if shared_max[tidx + stride] > shared_max[tidx]:
                shared_max[tidx] = shared_max[tidx + stride]
        cute.sync_threads()
    
    row_max = shared_max[0]
    cute.sync_threads()
    
    # Compute exp and sum
    sum_val = 0.0
    for col in range(tidx, N, 256):
        exp_val = cute.exp(gX[row * N + col] - row_max)
        gX[row * N + col] = exp_val
        sum_val += exp_val
    
    shared_sum[tidx] = sum_val
    cute.sync_threads()
    
    # Reduction for sum
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        if tidx < stride and tidx + stride < 256:
            shared_sum[tidx] += shared_sum[tidx + stride]
        cute.sync_threads()
    
    row_sum = shared_sum[0]
    cute.sync_threads()
    
    # Normalize
    for col in range(tidx, N, 256):
        gY[row * N + col] = gX[row * N + col] / row_sum

@cute.jit
def fused_gemm_gelu_softmax_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mOut: cute.Tensor,
    alpha: float, beta: float
):
    M, K = mA.shape
    K_B, N = mB.shape
    
    # Launch GEMM+GELU kernel
    grid_x = cute.ceil_div(N, 128)
    grid_y = cute.ceil_div(M, 128)
    
    gemm_gelu_softmax_kernel(mA, mB, mC, alpha, beta, M, N, K).launch(
        grid=(grid_x, grid_y, 1), 
        block=(128, 128, 1)
    )
    
    # Launch softmax kernel
    softmax_kernel(mA, mOut, M, N).launch(
        grid=(M, 1, 1),
        block=(256, 1, 1)
    )

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.compiled = {}
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.weight.t().contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(self.bias.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile kernel if not already compiled
        key = (x.dtype, batch_size, self.in_features, self.out_features)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_gelu_softmax_host, mA, mB, mC, mOut, 1.0, 1.0)
            self.compiled[key] = compiled
        
        # Launch kernel
        compiled(mA, mB, mC, mOut, 1.0, 1.0)
        
        return output