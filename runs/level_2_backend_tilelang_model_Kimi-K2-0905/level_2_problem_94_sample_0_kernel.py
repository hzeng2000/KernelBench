import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_gemm_bias_activation_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_gemm_bias_activation_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M

            local_A = T.alloc_fragment((block_M, block_K), dtype)
            local_B = T.alloc_fragment((block_N, block_K), dtype)
            local_C = T.alloc_fragment((block_M, block_N), dtype)

            for i, j in T.Parallel(block_M, block_N):
                local_C[i, j] = T.cast(0, dtype)

            for k in T.ceildiv(K, block_K):
                start_k = k * block_K

                for i, j in T.Parallel(block_M, block_K):
                    if start_m + i < M and start_k + j < K:
                        local_A[i, j] = A[start_m + i, start_k + j]
                    else:
                        local_A[i, j] = T.cast(0, dtype)

                for i, j in T.Parallel(block_N, block_K):
                    if start_n + i < N and start_k + j < K:
                        local_B[i, j] = B[start_n + i, start_k + j]
                    else:
                        local_B[i, j] = T.cast(0, dtype)

                for i, j in T.Parallel(block_M, block_N):
                    for kk in range(block_K):
                        local_C[i, j] += local_A[i, kk] * local_B[j, kk]

            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < M and start_n + j < N:
                    val = local_C[i, j] + bias[start_n + j]
                    # Hardtanh: clamp between -1 and 1
                    val = T.max(T.min(val, T.cast(1, dtype)), T.cast(-1, dtype))
                    # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
                    exp_val = T.exp(val)
                    softplus = T.log(T.cast(1, dtype) + exp_val)
                    tanh_sp = T.tanh(softplus)
                    C[start_m + i, start_n + j] = val * tanh_sp

    return tilelang.compile(fused_gemm_bias_activation_kernel, out_idx=[3], target="cuda")


def build_groupnorm_kernel(M: int, N: int, num_groups: int, block_M: int = 64, block_N: int = 64, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def groupnorm_kernel(
        X: T.Tensor((M, N), dtype),
        gamma: T.Tensor((N,), dtype),
        beta: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx,):
            start_m = bx * block_M
            
            channels_per_group = N // num_groups
            
            for local_m in T.Parallel(block_M):
                if start_m + local_m < M:
                    for g in range(num_groups):
                        # Compute mean
                        sum_val = T.cast(0, dtype)
                        for c in range(channels_per_group):
                            sum_val += X[start_m + local_m, g * channels_per_group + c]
                        mean = sum_val / T.cast(channels_per_group, dtype)
                        
                        # Compute variance
                        var_sum = T.cast(0, dtype)
                        for c in range(channels_per_group):
                            diff = X[start_m + local_m, g * channels_per_group + c] - mean
                            var_sum += diff * diff
                        var = var_sum / T.cast(channels_per_group, dtype)
                        
                        # Normalize
                        for c in range(channels_per_group):
                            idx = g * channels_per_group + c
                            normalized = (X[start_m + local_m, idx] - mean) / T.sqrt(var + T.cast(1e-5, dtype))
                            Y[start_m + local_m, idx] = normalized * gamma[idx] + beta[idx]

    return tilelang.compile(groupnorm_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # GroupNorm parameters
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        
        self._fused_kernel_cache = {}
        self._groupnorm_kernel_cache = {}

    def _get_fused_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._fused_kernel_cache:
            self._fused_kernel_cache[key] = build_fused_gemm_bias_activation_kernel(M, N, K, dtype=tl_dtype)
        return self._fused_kernel_cache[key]

    def _get_groupnorm_kernel(self, M: int, N: int, num_groups: int, tl_dtype: str):
        key = (M, N, num_groups, tl_dtype)
        if key not in self._groupnorm_kernel_cache:
            self._groupnorm_kernel_cache[key] = build_groupnorm_kernel(M, N, num_groups, dtype=tl_dtype)
        return self._groupnorm_kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        
        M, K = x_c.shape
        N = self.out_features
        
        # Convert to fp16
        x_fp16 = x_c.half()
        weight_fp16 = self.weight.t().half()  # Transpose for kernel
        bias_fp16 = self.bias.half()
        
        # Fused GEMM + Bias + Hardtanh + Mish
        fused_kernel = self._get_fused_kernel(M, N, K, "float16")
        intermediate = fused_kernel(x_fp16, weight_fp16, bias_fp16)
        
        # GroupNorm using PyTorch's implementation (as it's already optimized)
        output = self.groupnorm(intermediate)
        
        return output