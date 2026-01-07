import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_gemm_sigmoid_scaling_residual_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_sigmoid_scaling_residual_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
        scaling_factor: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            # Allocate shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            
            # Allocate local accumulator
            C_local = T.alloc_fragment((block_M, block_N), "float32", scope="local")
            
            # Initialize accumulator
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = 0.0
            
            # Main computation loop
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                # Load A tile to shared memory
                for i, k in T.Parallel(block_M, block_K):
                    global_i = start_y + i
                    global_k = ko * block_K + k
                    if global_i < M and global_k < K:
                        A_shared[i, k] = A[global_i, global_k]
                    else:
                        A_shared[i, k] = 0.0
                
                # Load B tile to shared memory (transposed)
                for j, k in T.Parallel(block_N, block_K):
                    global_j = start_x + j
                    global_k = ko * block_K + k
                    if global_j < N and global_k < K:
                        B_shared[j, k] = B[global_j, global_k]
                    else:
                        B_shared[j, k] = 0.0
                
                # Compute GEMM tile
                for ki in T.serial(block_K):
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += A_shared[i, ki].astype("float32") * B_shared[j, ki].astype("float32")
            
            # Store result and apply sigmoid, scaling, and residual
            for i, j in T.Parallel(block_M, block_N):
                global_i = start_y + i
                global_j = start_x + j
                if global_i < M and global_j < N:
                    # Apply sigmoid
                    sigmoid_val = 1.0 / (1.0 + T.exp(-C_local[i, j]))
                    # Scale and add residual
                    C[global_i, global_j] = (sigmoid_val * scaling_factor + C_local[i, j]).astype(dtype)

    return tilelang.compile(gemm_sigmoid_scaling_residual_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_sigmoid_scaling_residual_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        
        M = x_c.size(0)
        K = x_c.size(1)
        N = self.weight.size(0)
        
        # Convert to fp16
        x_fp16 = x_c.half()
        weight_fp16 = self.weight.half()
        
        kernel = self._get_kernel(M, N, K, "float16")
        output = kernel(x_fp16, weight_fp16, self.scaling_factor)
        
        # Add bias
        output = output + self.bias.half()
        
        return output