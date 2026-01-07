import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_gemm_bn_gelu_relu_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    @T.prim_func
    def gemm_bn_gelu_relu_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        running_mean: T.Tensor((N,), "float32"),
        running_var: T.Tensor((N,), "float32"),
        weight: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M
            
            acc = T.alloc_fragment((block_M, block_N), dtype)
            
            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] = T.cast(0.0, dtype)
            
            for k in range(0, K, block_K):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                
                for i, j in T.Parallel(block_M, block_K):
                    if start_m + i < M and k + j < K:
                        A_shared[i, j] = A[start_m + i, k + j]
                    else:
                        A_shared[i, j] = T.cast(0.0, dtype)
                
                for i, j in T.Parallel(block_N, block_K):
                    if start_n + i < N and k + j < K:
                        B_shared[i, j] = B[start_n + i, k + j]
                    else:
                        B_shared[i, j] = T.cast(0.0, dtype)
                
                for i, j, kk in T.Parallel(block_M, block_N, block_K):
                    acc[i, j] += T.cast(A_shared[i, kk] * B_shared[j, kk], dtype)
            
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < M and start_n + j < N:
                    # Add bias
                    val = acc[i, j] + bias[start_n + j]
                    
                    # BatchNorm (using running stats)
                    bn_val = (T.cast(val, "float32") - running_mean[start_n + j]) / T.sqrt(running_var[start_n + j] + T.cast(1e-5, "float32"))
                    bn_val = bn_val * weight[start_n + j]
                    
                    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    x = bn_val
                    x_cubed = x * x * x
                    tanh_arg = T.cast(math.sqrt(2.0 / math.pi), "float32") * (x + T.cast(0.044715, "float32") * x_cubed)
                    tanh_val = T.tanh(tanh_arg)
                    gelu_val = T.cast(0.5, "float32") * x * (T.cast(1.0, "float32") + tanh_val)
                    
                    # ReLU
                    relu_val = T.max(T.cast(0.0, "float32"), gelu_val)
                    
                    C[start_m + i, start_n + j] = T.cast(relu_val, dtype)
    
    return tilelang.compile(gemm_bn_gelu_relu_kernel, out_idx=[6], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self._kernel_cache = {}
        
    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_bn_gelu_relu_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]
    
    def forward(self, x):
        # Get weight and bias from linear layer
        weight = self.gemm.weight  # Shape: (out_features, in_features)
        bias = self.gemm.bias  # Shape: (out_features,)
        
        # Get BatchNorm parameters
        running_mean = self.batch_norm.running_mean  # Shape: (out_features,)
        running_var = self.batch_norm.running_var  # Shape: (out_features,)
        bn_weight = self.batch_norm.weight  # Shape: (out_features,)
        
        # Ensure tensors are contiguous and in fp16
        x_c = x.contiguous().half()
        weight_c = weight.contiguous().half()
        bias_c = bias.contiguous().half()
        
        M = x_c.size(0)
        N = weight_c.size(0)
        K = x_c.size(1)
        
        kernel = self._get_kernel(M, N, K, "float16")
        output = kernel(x_c, weight_c, bias_c, running_mean, running_var, bn_weight)
        
        return output