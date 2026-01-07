import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_gemm_subtract_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    @T.prim_func
    def gemm_subtract_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((N,)),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M

            local_A = T.alloc_fragment((block_M, block_K), dtype)
            local_B = T.alloc_fragment((block_N, block_K), dtype)
            local_C = T.alloc_fragment((block_M, block_N), dtype)

            for i, j in T.Parallel(block_M, block_N):
                local_C[i, j] = T.float16(0.0)

            for k in range(0, K, block_K):
                for i, j in T.Parallel(block_M, block_K):
                    if start_m + i < M and k + j < K:
                        local_A[i, j] = A[start_m + i, k + j]
                    else:
                        local_A[i, j] = T.float16(0.0)

                for i, j in T.Parallel(block_N, block_K):
                    if start_n + i < N and k + j < K:
                        local_B[i, j] = B[start_n + i, k + j]
                    else:
                        local_B[i, j] = T.float16(0.0)

                for i, j in T.Parallel(block_M, block_N):
                    for kk in range(block_K):
                        local_C[i, j] += local_A[i, kk] * local_B[j, kk]

            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < M and start_n + j < N:
                    D[start_m + i, start_n + j] = local_C[i, j] - C[start_n + j]

    return tilelang.compile(gemm_subtract_kernel, out_idx=[3], target="cuda")


def build_reduce_gelu_kernel(M: int, N: int, block_M: int = 64, threads: int = 256, dtype: str = "float16"):
    @T.prim_func
    def reduce_gelu_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx,):
            start_m = bx * block_M

            local_sum = T.alloc_fragment((block_M,), dtype)
            local_max = T.alloc_fragment((block_M,), "float16")

            for i in T.Parallel(block_M):
                local_sum[i] = T.float16(0.0)
                local_max[i] = T.float16(-1e4)

            # Compute mean
            for i in T.Parallel(block_M):
                for j in range(N):
                    if start_m + i < M:
                        local_sum[i] += A[start_m + i, j]

            for i in T.Parallel(block_M):
                if start_m + i < M:
                    mean_val = local_sum[i] / T.float16(N)
                    for j in range(N):
                        C[start_m + i, j] = mean_val

            # Compute logsumexp
            for i in T.Parallel(block_M):
                local_max[i] = T.float16(-1e4)
                for j in range(N):
                    if start_m + i < M:
                        local_max[i] = T.max(local_max[i], C[start_m + i, j])

            for i in T.Parallel(block_M):
                if start_m + i < M:
                    max_val = local_max[i]
                    exp_sum = T.float16(0.0)
                    for j in range(N):
                        exp_val = T.exp(C[start_m + i, j] - max_val)
                        exp_sum += exp_val

                    logsumexp_val = T.log(exp_sum) + max_val
                    for j in range(N):
                        C[start_m + i, j] = logsumexp_val

            # GELU
            for i, j in T.Parallel(block_M, N):
                if start_m + i < M:
                    x = C[start_m + i, j]
                    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    pi = T.float16(3.141592653589793)
                    sqrt_2_over_pi = T.sqrt(T.float16(2.0) / pi)
                    x_cubed = x * x * x
                    tanh_arg = sqrt_2_over_pi * (x + T.float16(0.044715) * x_cubed)
                    # Approximate tanh
                    tanh_approx = tanh_arg / (T.abs(tanh_arg) + T.float16(1.0))
                    gelu_val = T.float16(0.5) * x * (T.float16(1.0) + tanh_approx)
                    C[start_m + i, j] = gelu_val + B[start_m + i, j]

    return tilelang.compile(reduce_gelu_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self._kernel_cache = {}

    def _get_gemm_subtract_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = ("gemm_subtract", M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_subtract_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_reduce_gelu_kernel(self, M: int, N: int, tl_dtype: str):
        key = ("reduce_gelu", M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_reduce_gelu_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        original_x = x.clone().detach()
        
        M, K = x.shape
        N = self.gemm.out_features
        
        # Convert to FP16
        x_fp16 = x.half()
        weight_fp16 = self.gemm.weight.half().t()
        bias_fp16 = self.subtract.half()
        
        # Gemm + Subtract
        kernel1 = self._get_gemm_subtract_kernel(M, N, K, "float16")
        x_out = kernel1(x_fp16, weight_fp16, bias_fp16)
        
        # GlobalAvgPool + LogSumExp + GELU + ResidualAdd
        kernel2 = self._get_reduce_gelu_kernel(M, N, "float16")
        x_final = kernel2(x_out, original_x.half())
        
        return x_final.float()