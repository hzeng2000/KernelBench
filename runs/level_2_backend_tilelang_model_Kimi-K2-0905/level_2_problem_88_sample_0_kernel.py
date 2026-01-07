import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_gemm_swish_gn_mul_swish_kernel(
    M: int, N: int, K: int, num_groups: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"
):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        gn_weight: T.Tensor((num_groups,), dtype),
        gn_bias: T.Tensor((num_groups,), dtype),
        mul_weight: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M

            # Allocate shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            # Initialize C_local to zero
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.cast(0, dtype)

            # Loop over K dimension
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                start_k = k * block_K

                # Load A into shared memory
                for i, j in T.Parallel(block_M, block_K):
                    if start_m + i < M and start_k + j < K:
                        A_shared[i, j] = A[start_m + i, start_k + j]
                    else:
                        A_shared[i, j] = T.cast(0, dtype)

                # Load B into shared memory
                for i, j in T.Parallel(block_N, block_K):
                    if start_n + i < N and start_k + j < K:
                        B_shared[i, j] = B[start_n + i, start_k + j]
                    else:
                        B_shared[i, j] = T.cast(0, dtype)

                # Compute GEMM
                for i, j in T.Parallel(block_M, block_N):
                    for kk in range(block_K):
                        C_local[i, j] += A_shared[i, kk] * B_shared[j, kk]

            # Apply bias
            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < M and start_n + j < N:
                    C_local[i, j] += bias[start_n + j]

            # GroupNorm + Swish + Mul + Swish fused
            group_size = N // num_groups
            for g in range(num_groups):
                # Compute mean and inv_std for this group
                mean = T.alloc_fragment((1,), "float32")
                mean[0] = T.cast(0, "float32")
                for j in range(group_size):
                    col = g * group_size + j
                    if start_n + col < N:
                        for i in range(block_M):
                            if start_m + i < M:
                                mean[0] += T.cast(C_local[i, col], "float32")
                mean[0] /= T.cast(M * group_size, "float32")

                var = T.alloc_fragment((1,), "float32")
                var[0] = T.cast(0, "float32")
                for j in range(group_size):
                    col = g * group_size + j
                    if start_n + col < N:
                        for i in range(block_M):
                            if start_m + i < M:
                                diff = T.cast(C_local[i, col], "float32") - mean[0]
                                var[0] += diff * diff
                var[0] /= T.cast(M * group_size, "float32")
                inv_std = T.rsqrt(var[0] + T.cast(1e-5, "float32"))

                # Apply GroupNorm, Swish, Mul, Swish
                for j in range(group_size):
                    col = g * group_size + j
                    if start_n + col < N:
                        for i in range(block_M):
                            if start_m + i < M:
                                # GroupNorm
                                val = T.cast(C_local[i, col], "float32")
                                val = (val - mean[0]) * inv_std
                                val = val * T.cast(gn_weight[g], "float32") + T.cast(gn_bias[g], "float32")
                                val = T.cast(val, dtype)
                                # Swish
                                sig = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-val))
                                val = val * sig
                                # Multiply
                                val = val * mul_weight[start_n + col]
                                # Swish again
                                sig = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-val))
                                val = val * sig
                                C[start_m + i, start_n + col] = val

    return tilelang.compile(kernel, out_idx=[6], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features))
        self.gn_weight = nn.Parameter(torch.ones(num_groups))
        self.gn_bias = nn.Parameter(torch.zeros(num_groups))
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, num_groups: int):
        key = (M, N, K, num_groups)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_swish_gn_mul_swish_kernel(M, N, K, num_groups, dtype="float16")
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        kernel = self._get_kernel(M, N, K, self.num_groups)
        x_fp16 = x.half()
        weight_fp16 = self.gemm_weight.t().half()
        bias_fp16 = self.gemm_bias.half()
        gn_weight_fp16 = self.gn_weight.half()
        gn_bias_fp16 = self.gn_bias.half()
        mul_weight_fp16 = self.multiply_weight.half()
        out = kernel(x_fp16, weight_fp16, bias_fp16, gn_weight_fp16, gn_bias_fp16, mul_weight_fp16)
        return out