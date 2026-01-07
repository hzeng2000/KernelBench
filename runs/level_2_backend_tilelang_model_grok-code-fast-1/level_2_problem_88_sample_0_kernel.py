import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_matmul_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def matmul_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M
            for local_m, local_n in T.Parallel(block_M, block_N):
                m = start_m + local_m
                n = start_n + local_n
                if m < M and n < N:
                    sum = B[n]
                    for k in range(K):
                        sum += X[m, k] * W[n, k]
                    Y[m, n] = sum
    return tilelang.compile(matmul_kernel, out_idx=[3], target="cuda")


def build_fused_groupnorm_elementwise_kernel(N: int, C: int, G: int, block_N: int = 32, threads: int = 32, dtype: str = "float16"):
    @T.prim_func
    def fused_groupnorm_elementwise_kernel(
        X: T.Tensor((N, C), dtype),
        gn_weight: T.Tensor((C,), dtype),
        gn_bias: T.Tensor((C,), dtype),
        mul_weight: T.Tensor((C,), dtype),
        Y: T.Tensor((N, C), dtype),
    ):
        fg = C // G
        eps = 1e-5
        with T.Kernel(T.ceildiv(N, block_N), G, threads=threads) as (bx, g):
            n = bx * block_N + T.thread_binding(0, block_N, "threadIdx.x")
            if n < N:
                sum_val = 0.0
                sum_sq = 0.0
                for k in range(fg):
                    c = g * fg + k
                    val = X[n, c]
                    sum_val += val
                    sum_sq += val * val
                mean = sum_val / fg
                var = sum_sq / fg - mean * mean
                for k in range(fg):
                    c = g * fg + k
                    z = (X[n, c] - mean) / T.sqrt(var + eps)
                    z = z * gn_weight[c] + gn_bias[c]
                    sig = 1 / (1 + T.exp(-z))
                    z = z * sig
                    z = z * mul_weight[c]
                    sig = 1 / (1 + T.exp(-z))
                    Y[n, c] = z * sig
    return tilelang.compile(fused_groupnorm_elementwise_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self._matmul_kernel_cache = {}
        self._fused_kernel_cache = {}

    def _get_matmul_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._matmul_kernel_cache:
            self._matmul_kernel_cache[key] = build_matmul_kernel(M, K, N, dtype=tl_dtype)
        return self._matmul_kernel_cache[key]

    def _get_fused_kernel(self, N: int, C: int, G: int, tl_dtype: str):
        key = (N, C, G, tl_dtype)
        if key not in self._fused_kernel_cache:
            self._fused_kernel_cache[key] = build_fused_groupnorm_elementwise_kernel(N, C, G, dtype=tl_dtype)
        return self._fused_kernel_cache[key]

    def forward(self, x):
        batch_size, in_features = x.shape
        out_features = self.gemm.out_features
        num_groups = self.group_norm.num_groups
        kernel_matmul = self._get_matmul_kernel(batch_size, in_features, out_features, "float16")
        x = kernel_matmul(x, self.gemm.weight, self.gemm.bias)
        kernel_fused = self._get_fused_kernel(batch_size, out_features, num_groups, "float16")
        x = kernel_fused(x, self.group_norm.weight, self.group_norm.bias, self.multiply_weight)
        return x