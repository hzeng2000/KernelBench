import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemm_max_kernel(batch_size: int, in_features: int, out_features: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_max_kernel(
        A: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        C: T.Tensor((batch_size, out_features), dtype),
        Max: T.Tensor((batch_size, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), threads=threads) as (bx):
            start_m = bx * block_M
            
            # Shared memory for input tile
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            # Register for output tile
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            # Register for max values
            max_local = T.alloc_fragment((block_M,), dtype)
            
            # Initialize max values
            for i in T.Parallel(block_M):
                max_local[i] = T.min_value(dtype)
            
            # Compute GEMM and find max per row
            for ko in range(T.ceildiv(in_features, block_K)):
                # Load input tile
                for i, k in T.Parallel(block_M, block_K):
                    if start_m + i < batch_size and ko * block_K + k < in_features:
                        A_shared[i, k] = A[start_m + i, ko * block_K + k]
                    else:
                        A_shared[i, k] = T.cast(0, dtype)
                
                for no in range(T.ceildiv(out_features, block_N)):
                    # Initialize output tile
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] = T.cast(0, dtype)
                    
                    # Compute tile
                    for ki in range(block_K):
                        for i, j in T.Parallel(block_M, block_N):
                            if start_m + i < batch_size and no * block_N + j < out_features:
                                if ki == 0:
                                    C_local[i, j] = T.cast(B[no * block_N + j], dtype)
                                if ko * block_K + ki < in_features:
                                    C_local[i, j] += A_shared[i, ki] * W[no * block_N + j, ko * block_K + ki]
                    
                    # Store output and update max
                    for i, j in T.Parallel(block_M, block_N):
                        if start_m + i < batch_size and no * block_N + j < out_features:
                            C[start_m + i, no * block_N + j] = C_local[i, j]
                            if C_local[i, j] > max_local[i]:
                                max_local[i] = C_local[i, j]
            
            # Store max values
            for i in T.Parallel(block_M):
                if start_m + i < batch_size:
                    Max[start_m + i, 0] = max_local[i]

    return tilelang.compile(gemm_max_kernel, out_idx=[3, 4], target="cuda")


def build_sub_gelu_kernel(batch_size: int, out_features: int, block_M: int = 64, block_N: int = 64, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def sub_gelu_kernel(
        C: T.Tensor((batch_size, out_features), dtype),
        Max: T.Tensor((batch_size, 1), dtype),
        Out: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), T.ceildiv(out_features, block_N), threads=threads) as (bx, by):
            start_m = bx * block_M
            start_n = by * block_N
            
            # Shared memory for mean computation
            mean_shared = T.alloc_shared((block_M,), dtype)
            # Register for tile
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            # Load data and subtract max
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < batch_size and start_n + j < out_features:
                    C_local[i, j] = C[start_m + i, start_n + j] - Max[start_m + i, 0]
            
            # Compute mean (simplified - assume we can use a reduction)
            for i in T.Parallel(block_M):
                if start_m + i < batch_size:
                    sum_val = T.cast(0, dtype)
                    for j in range(out_features):
                        sum_val += C[start_m + i, j] - Max[start_m + i, 0]
                    mean_shared[i] = sum_val / T.cast(out_features, dtype)
            
            # Apply subtraction and GELU
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < batch_size and start_n + j < out_features:
                    val = C_local[i, j] - mean_shared[i]
                    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    x = val
                    pi = T.cast(3.14159265359, dtype)
                    sqrt_2_over_pi = T.sqrt(T.cast(2, dtype) / pi)
                    x_cubed = x * x * x
                    tanh_arg = sqrt_2_over_pi * (x + T.cast(0.044715, dtype) * x_cubed)
                    tanh_val = T.tanh(tanh_arg)
                    Out[start_m + i, start_n + j] = T.cast(0.5, dtype) * x * (T.cast(1, dtype) + tanh_val)

    return tilelang.compile(sub_gelu_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self._kernel_cache = {}

    def _get_gemm_max_kernel(self, batch_size: int, in_features: int, out_features: int):
        key = ("gemm_max", batch_size, in_features, out_features)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_max_kernel(batch_size, in_features, out_features)
        return self._kernel_cache[key]

    def _get_sub_gelu_kernel(self, batch_size: int, out_features: int):
        key = ("sub_gelu", batch_size, out_features)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_sub_gelu_kernel(batch_size, out_features)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = self.gemm.out_features
        
        # Ensure FP16
        x = x.half()
        
        # Get weight and bias
        weight = self.gemm.weight.t().half()
        bias = self.gemm.bias.half()
        
        # Run fused GEMM + Max kernel
        gemm_max_kernel = self._get_gemm_max_kernel(batch_size, in_features, out_features)
        x_gemm, x_max = gemm_max_kernel(x, weight, bias)
        
        # Run fused Sub + GELU kernel
        sub_gelu_kernel = self._get_sub_gelu_kernel(batch_size, out_features)
        output = sub_gelu_kernel(x_gemm, x_max)
        
        return output