import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_fused_gemm_logsumexp_leaky_gelu_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M
            
            # Shared memory for tiles
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            
            # Local accumulators
            acc = T.alloc_fragment((block_M, block_N), dtype)
            max_vals = T.alloc_fragment((block_M,), dtype)
            sum_vals = T.alloc_fragment((block_M,), dtype)
            
            # Initialize
            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] = T.cast(0, dtype)
                if j == 0:
                    max_vals[i] = T.cast(-1e10, dtype)
                    sum_vals[i] = T.cast(0, dtype)
            
            # Main GEMM loop
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                # Load A tile
                for i, j in T.Parallel(block_M, block_K):
                    if start_m + i < M and k * block_K + j < K:
                        A_shared[i, j] = A[start_m + i, k * block_K + j]
                    else:
                        A_shared[i, j] = T.cast(0, dtype)
                
                # Load B tile
                for i, j in T.Parallel(block_N, block_K):
                    if start_n + i < N and k * block_K + j < K:
                        B_shared[i, j] = B[start_n + i, k * block_K + j]
                    else:
                        B_shared[i, j] = T.cast(0, dtype)
                
                # Compute GEMM tile
                for i, j, kk in T.Parallel(block_M, block_N, block_K):
                    acc[i, j] += A_shared[i, kk] * B_shared[j, kk]
            
            # Add bias and compute logsumexp
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < M and start_n + j < N:
                    val = acc[i, j] + bias[start_n + j]
                    # First LeakyReLU
                    val = T.select(val > 0, val, val * T.cast(0.01, dtype))
                    # Second LeakyReLU
                    val = T.select(val > 0, val, val * T.cast(0.01, dtype))
                    # First GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    x_cubed = val * val * val
                    tanh_arg = T.cast(0.7978845608, dtype) * (val + T.cast(0.044715, dtype) * x_cubed)
                    # Approximate tanh with a rational function
                    tanh_approx = tanh_arg / (T.cast(1.0, dtype) + T.abs(tanh_arg))
                    gelu1 = T.cast(0.5, dtype) * val * (T.cast(1.0, dtype) + tanh_approx)
                    # Second GELU
                    x_cubed2 = gelu1 * gelu1 * gelu1
                    tanh_arg2 = T.cast(0.7978845608, dtype) * (gelu1 + T.cast(0.044715, dtype) * x_cubed2)
                    tanh_approx2 = tanh_arg2 / (T.cast(1.0, dtype) + T.abs(tanh_arg2))
                    final_val = T.cast(0.5, dtype) * gelu1 * (T.cast(1.0, dtype) + tanh_approx2)
                    
                    # Update max and sum for logsumexp
                    if j == 0:
                        max_vals[i] = final_val
                        sum_vals[i] = T.cast(1.0, dtype)
                    else:
                        old_max = max_vals[i]
                        new_max = T.max(old_max, final_val)
                        exp_diff = T.exp(old_max - new_max)
                        exp_val = T.exp(final_val - new_max)
                        sum_vals[i] = sum_vals[i] * exp_diff + exp_val
                        max_vals[i] = new_max
            
            # Final logsumexp result
            for i in T.Parallel(block_M):
                if start_m + i < M:
                    C[start_m + i, 0] = T.log(sum_vals[i]) + max_vals[i]
    
    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._kernel_cache = {}
        
    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_gemm_logsumexp_leaky_gelu_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get weight and bias
        weight = self.linear.weight  # Shape: (out_features, in_features)
        bias = self.linear.bias      # Shape: (out_features,)
        
        # Ensure tensors are contiguous and in fp16
        x = x.contiguous().half()
        weight = weight.contiguous().half()
        if bias is not None:
            bias = bias.contiguous().half()
        else:
            bias = torch.zeros(weight.size(0), dtype=torch.float16, device=x.device)
        
        batch_size, in_features = x.shape
        out_features = weight.size(0)
        
        # Get kernel
        kernel = self._get_kernel(batch_size, out_features, in_features, "float16")
        
        # Allocate output
        output = torch.empty(batch_size, 1, dtype=torch.float16, device=x.device)
        
        # Launch kernel
        kernel(x, weight, bias, output)
        
        return output