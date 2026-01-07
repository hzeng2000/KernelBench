import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_matmul_gelu_softmax_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def matmul_gelu_softmax_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Shared memory for tiles
        shared_A = T.alloc_shared((block_M, block_K), dtype)
        shared_B = T.alloc_shared((block_N, block_K), dtype)
        
        # Local accumulators
        local_C = T.alloc_fragment((block_M, block_N), "float32")
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M
            
            # Initialize local accumulators
            for i, j in T.Parallel(block_M, block_N):
                local_C[i, j] = 0.0
            
            # Loop over K dimension
            for k_tile in range(T.ceildiv(K, block_K)):
                start_k = k_tile * block_K
                
                # Load A tile to shared memory
                for i, k in T.Parallel(block_M, block_K):
                    global_m = start_m + i
                    global_k = start_k + k
                    if global_m < M and global_k < K:
                        shared_A[i, k] = A[global_m, global_k]
                    else:
                        shared_A[i, k] = 0.0
                
                # Load B tile to shared memory (transposed)
                for j, k in T.Parallel(block_N, block_K):
                    global_n = start_n + j
                    global_k = start_k + k
                    if global_n < N and global_k < K:
                        shared_B[j, k] = B[global_n, global_k]
                    else:
                        shared_B[j, k] = 0.0
                
                # Compute matmul for this tile
                for i, j, k in T.Parallel(block_M, block_N, block_K):
                    local_C[i, j] += shared_A[i, k] * shared_B[j, k]
            
            # Apply GELU and online softmax
            # First, compute row-wise max for numerical stability
            row_max = T.alloc_fragment((block_M,), "float32")
            for i in T.Parallel(block_M):
                row_max[i] = -float('inf')
                for j in range(block_N):
                    global_m = start_m + i
                    global_n = start_n + j
                    if global_m < M and global_n < N:
                        val = local_C[i, j]
                        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        gelu_val = 0.5 * val * (1.0 + T.tanh(0.7978845608 * (val + 0.044715 * val * val * val)))
                        if gelu_val > row_max[i]:
                            row_max[i] = gelu_val
            
            # Compute exp and sum
            row_sum = T.alloc_fragment((block_M,), "float32")
            for i in T.Parallel(block_M):
                row_sum[i] = 0.0
                for j in range(block_N):
                    global_m = start_m + i
                    global_n = start_n + j
                    if global_m < M and global_n < N:
                        val = local_C[i, j]
                        # GELU approximation
                        gelu_val = 0.5 * val * (1.0 + T.tanh(0.7978845608 * (val + 0.044715 * val * val * val)))
                        exp_val = T.exp(gelu_val - row_max[i])
                        row_sum[i] += exp_val
            
            # Write normalized softmax output
            for i, j in T.Parallel(block_M, block_N):
                global_m = start_m + i
                global_n = start_n + j
                if global_m < M and global_n < N:
                    val = local_C[i, j]
                    # GELU approximation
                    gelu_val = 0.5 * val * (1.0 + T.tanh(0.7978845608 * (val + 0.044715 * val * val * val)))
                    exp_val = T.exp(gelu_val - row_max[i])
                    C[global_m, global_n] = exp_val / row_sum[i]
    
    return tilelang.compile(matmul_gelu_softmax_kernel, out_idx=[2], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self._kernel_cache = {}

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_matmul_gelu_softmax_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        
        batch_size = x_c.size(0)
        M, K, N = batch_size, self.in_features, self.out_features
        
        # Apply linear transformation manually
        x_reshaped = x_c.view(-1, K)
        weight_t = self.weight.t().contiguous()
        
        kernel = self._get_kernel(M, N, K, "float16")
        
        # Allocate output tensor
        output = torch.empty(M, N, dtype=torch.float16, device=x.device)
        
        # Call fused kernel
        kernel(x_reshaped.to(torch.float16), weight_t.to(torch.float16), output)
        
        # Add bias
        output = output + self.bias.to(torch.float16).unsqueeze(0)
        
        return output