import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_gemm_groupnorm_min_bias_kernel(
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
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
        weight: T.Tensor((N,), dtype),
        bias: T.Tensor((N,), dtype),
        bias_add: T.Tensor((1, N, 1, 1), dtype),
    ):
        # Shared memory for tiling
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_N, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), "float32")
        
        # Group norm parameters per group
        group_size = N // num_groups
        mean_shared = T.alloc_shared((block_M, num_groups), "float32")
        var_shared = T.alloc_shared((block_M, num_groups), "float32")
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_m = by * block_M
            start_n = bx * block_N
            
            # Initialize C_local
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = 0.0
            
            # GEMM computation
            for k in range(T.ceildiv(K, block_K)):
                # Load A tile
                for i, kk in T.Parallel(block_M, block_K):
                    if start_m + i < M and k * block_K + kk < K:
                        A_shared[i, kk] = A[start_m + i, k * block_K + kk]
                    else:
                        A_shared[i, kk] = 0.0
                
                # Load B tile
                for j, kk in T.Parallel(block_N, block_K):
                    if start_n + j < N and k * block_K + kk < K:
                        B_shared[j, kk] = B[start_n + j, k * block_K + kk]
                    else:
                        B_shared[j, kk] = 0.0
                
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            # GroupNorm + Min + Bias fusion
            for i in T.Parallel(block_M):
                if start_m + i < M:
                    # Compute group statistics
                    for g in range(num_groups):
                        sum_val = 0.0
                        for j in range(group_size):
                            idx = g * group_size + j
                            if start_n + idx < N:
                                sum_val += C_local[i, idx]
                        mean_shared[i, g] = sum_val / group_size
                    
                    # Compute variance
                    for g in range(num_groups):
                        var_sum = 0.0
                        for j in range(group_size):
                            idx = g * group_size + j
                            if start_n + idx < N:
                                diff = C_local[i, idx] - mean_shared[i, g]
                                var_sum += diff * diff
                        var_shared[i, g] = var_sum / group_size + 1e-5
                    
                    # Apply GroupNorm
                    for j in range(block_N):
                        if start_n + j < N:
                            g = j // group_size
                            normalized = (C_local[i, j] - mean_shared[i, g]) / T.sqrt(var_shared[i, g])
                            C_local[i, j] = normalized * weight[start_n + j] + bias[start_n + j]
                    
                    # Find min across N dimension for this row
                    min_val = C_local[i, 0]
                    for j in range(1, block_N):
                        if start_n + j < N and C_local[i, j] < min_val:
                            min_val = C_local[i, j]
                    
                    # Add bias and store
                    for j in range(block_N):
                        if start_n + j < N:
                            C[start_m + i, start_n + j] = min_val + bias_add[0, start_n + j, 0, 0]

    return tilelang.compile(kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_linear = nn.Parameter(torch.randn(out_features))
        self.group_norm_weight = nn.Parameter(torch.ones(out_features))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_features))
        self.bias_add = nn.Parameter(torch.randn(bias_shape))
        
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int, num_groups: int):
        key = (M, K, N, num_groups)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_groupnorm_min_bias_kernel(
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
        gn_weight_fp16 = self.group_norm_weight.half()
        gn_bias_fp16 = self.group_norm_bias.half()
        bias_add_fp16 = self.bias_add.half()
        
        kernel = self._get_kernel(M, K, N, self.num_groups)
        output = kernel(x_fp16, weight_fp16, gn_weight_fp16, gn_bias_fp16, bias_add_fp16)
        
        return output.view(M, 1, 1, N).expand(-1, 1, 1, -1)