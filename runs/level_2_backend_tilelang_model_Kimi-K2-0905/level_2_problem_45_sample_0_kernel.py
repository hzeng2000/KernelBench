import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_gemm_sigmoid_gemm_lse_kernel(M: int, K1: int, N1: int, K2: int, N2: int, 
                                       block_M: int = 64, block_N: int = 64, block_K: int = 32,
                                       threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K1), dtype),
        B1: T.Tensor((K1, N1), dtype),
        B2: T.Tensor((N1, N2), dtype),
        C: T.Tensor((M,), dtype),
    ):
        # Shared memory allocations
        shared_A = T.alloc_shared((block_M, block_K), dtype)
        shared_B1 = T.alloc_shared((block_K, block_N), dtype)
        shared_B2 = T.alloc_shared((block_N, block_K), dtype)
        
        # Register allocations
        local_acc1 = T.alloc_fragment((block_M, block_N), dtype)
        local_acc2 = T.alloc_fragment((block_M, block_N), dtype)
        local_max = T.alloc_fragment((block_M,), dtype)
        local_sum = T.alloc_fragment((block_M,), dtype)
        
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            start_m = bx * block_M
            
            # Initialize max and sum for logsumexp
            for i in T.Parallel(block_M):
                local_max[i] = T.min_value(dtype)
                local_sum[i] = T.cast(0.0, dtype)
            
            # First GEMM: A @ B1
            for k1 in T.range(0, K1, block_K):
                # Load A tile
                for i, j in T.Parallel(block_M, block_K):
                    global_i = start_m + i
                    global_j = k1 + j
                    if global_i < M and global_j < K1:
                        shared_A[i, j] = A[global_i, global_j]
                    else:
                        shared_A[i, j] = T.cast(0.0, dtype)
                
                # Load B1 tile
                for i, j in T.Parallel(block_K, block_N):
                    global_i = k1 + i
                    global_j = j
                    if global_i < K1 and global_j < N1:
                        shared_B1[i, j] = B1[global_i, global_j]
                    else:
                        shared_B1[i, j] = T.cast(0.0, dtype)
                
                # Compute GEMM tile
                for i, j in T.Parallel(block_M, block_N):
                    local_acc1[i, j] = T.cast(0.0, dtype)
                    for k in T.serial(block_K):
                        local_acc1[i, j] += shared_A[i, k] * shared_B1[k, j]
            
            # Apply Sigmoid
            for i, j in T.Parallel(block_M, block_N):
                global_i = start_m + i
                global_j = j
                if global_i < M and global_j < N1:
                    val = local_acc1[i, j]
                    # Sigmoid approximation: 1 / (1 + exp(-x))
                    sigmoid = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-val))
                    local_acc1[i, j] = sigmoid
            
            # Second GEMM: sigmoid_result @ B2
            for k2 in T.range(0, N1, block_N):
                # Load B2 tile (transposed for memory coalescing)
                for i, j in T.Parallel(block_N, block_K):
                    global_i = k2 + i
                    global_j = j
                    if global_i < N1 and global_j < N2:
                        shared_B2[i, j] = B2[global_i, global_j]
                    else:
                        shared_B2[i, j] = T.cast(0.0, dtype)
                
                # Compute GEMM tile
                for i, j in T.Parallel(block_M, block_K):
                    local_acc2[i, j] = T.cast(0.0, dtype)
                    for k in T.serial(block_N):
                        if k2 + k < N1:
                            local_acc2[i, j] += local_acc1[i, k] * shared_B2[k, j]
            
            # LogSumExp computation
            for i in T.Parallel(block_M):
                global_i = start_m + i
                if global_i < M:
                    # Find max for numerical stability
                    max_val = T.min_value(dtype)
                    for j in T.serial(N2):
                        if local_acc2[i, j] > max_val:
                            max_val = local_acc2[i, j]
                    local_max[i] = max_val
                    
                    # Compute sum of exp(x - max)
                    sum_val = T.cast(0.0, dtype)
                    for j in T.serial(N2):
                        sum_val += T.exp(local_acc2[i, j] - max_val)
                    local_sum[i] = sum_val
                    
                    # Final logsumexp: max + log(sum)
                    C[global_i] = max_val + T.log(sum_val)
    
    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self._kernel_cache = {}
        
    def _get_kernel(self, M: int, K1: int, N1: int, K2: int, N2: int):
        key = (M, K1, N1, K2, N2)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_sigmoid_gemm_lse_kernel(M, K1, N1, K2, N2)
        return self._kernel_cache[key]
    
    def forward(self, x):
        M = x.shape[0]
        K1 = x.shape[1]
        N1 = self.linear1.out_features
        K2 = self.linear2.in_features
        N2 = self.linear2.out_features
        
        # Get weights
        W1 = self.linear1.weight.t().contiguous().half()
        W2 = self.linear2.weight.t().contiguous().half()
        
        # Get kernel
        kernel = self._get_kernel(M, K1, N1, K2, N2)
        
        # Run fused kernel
        output = kernel(x.half(), W1, W2)
        
        return output