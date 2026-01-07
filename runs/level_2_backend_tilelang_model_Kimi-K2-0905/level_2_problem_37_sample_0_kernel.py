import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_matmul_swish_add_gn_kernel(
    M: int, 
    N: int, 
    K: int, 
    num_groups: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        weight: T.Tensor((num_groups,), dtype),
        bias_gn: T.Tensor((num_groups,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M
            
            # Allocate shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype, accum=True)
            
            # Initialize accumulator
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.cast(0, dtype)
            
            # Compute number of K tiles
            num_k_tiles = T.ceildiv(K, block_K)
            
            for k_tile in range(num_k_tiles):
                start_k = k_tile * block_K
                
                # Load A tile to shared memory
                for i, k in T.Parallel(block_M, block_K):
                    global_m = start_m + i
                    global_k = start_k + k
                    if global_m < M and global_k < K:
                        A_shared[i, k] = A[global_m, global_k]
                    else:
                        A_shared[i, k] = T.cast(0, dtype)
                
                # Load B tile to shared memory (transposed)
                for j, k in T.Parallel(block_N, block_K):
                    global_n = start_n + j
                    global_k = start_k + k
                    if global_n < N and global_k < K:
                        B_shared[j, k] = B[global_n, global_k]
                    else:
                        B_shared[j, k] = T.cast(0, dtype)
                
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            # Apply Swish, add bias, and GroupNorm
            group_size = N // num_groups
            
            for i, j in T.Parallel(block_M, block_N):
                global_m = start_m + i
                global_n = start_n + j
                
                if global_m < M and global_n < N:
                    # Apply Swish activation
                    val = C_local[i, j]
                    sigmoid = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-val))
                    swish_val = val * sigmoid
                    
                    # Add bias
                    biased_val = swish_val + bias[global_n]
                    
                    # GroupNorm
                    group_idx = global_n // group_size
                    normalized_val = (biased_val - bias_gn[group_idx]) * weight[group_idx]
                    
                    C[global_m, global_n] = normalized_val
    
    return tilelang.compile(fused_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # GroupNorm parameters
        self.group_norm_weight = nn.Parameter(torch.ones(num_groups))
        self.group_norm_bias = nn.Parameter(torch.zeros(num_groups))
        
        self._kernel_cache = {}
        
    def _get_kernel(self, M: int, K: int, N: int, num_groups: int):
        key = (M, K, N, num_groups)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_swish_add_gn_kernel(
                M, N, K, num_groups, dtype="float16"
            )
        return self._kernel_cache[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = self.weight.half()
        bias_fp16 = self.bias.half()
        gn_weight_fp16 = self.group_norm_weight.half()
        gn_bias_fp16 = self.group_norm_bias.half()
        
        kernel = self._get_kernel(M, K, N, self.num_groups)
        output = kernel(x_fp16, weight_fp16, bias_fp16, gn_weight_fp16, gn_bias_fp16)
        
        return output.float()