import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_gemm_bn_scale(M: int, K: int, N: int, eps: float, block_M: int = 128, block_K: int = 64, block_N: int = 128, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def fused_gemm_bn_scale_kernel(
        A: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        running_mean: T.Tensor((N,), "float32"),
        running_var: T.Tensor((N,), "float32"),
        gamma: T.Tensor((N,), dtype),
        beta: T.Tensor((N,), dtype),
        scale: T.Tensor((), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M:, k * block_K:], A_shared)
                T.copy(W[bx * block_N:, k * block_K:], W_shared)
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M:, bx * block_N:])
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N:
                    val = C[m, n] + bias[n]
                    val = (val - running_mean[n]) / T.sqrt(running_var[n] + eps) * gamma[n] + beta[n]
                    C[m, n] = scale[()] * val
    return tilelang.compile(fused_gemm_bn_scale_kernel, out_idx=[8], target="cuda")


def build_row_max_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def row_max_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            max_shared = T.alloc_shared((block_M,), dtype)
            for i in T.Parallel(block_M):
                max_shared[i] = T.cast(-float('inf'), dtype)
            for k in T.Pipelined(T.ceildiv(N, block_N), num_stages=1):
                A_shared = T.alloc_shared((block_M, block_N), dtype)
                T.copy(A[bx * block_M:, k * block_N:], A_shared)
                for i in T.Parallel(block_M):
                    local_max = T.reduce(T.max, A_shared[i, :], axis=0, init=T.cast(-float('inf'), dtype))
                    max_shared[i] = T.max(max_shared[i], local_max)
            for i in T.Parallel(block_M):
                B[bx * block_M + i] = max_shared[i]
    return tilelang.compile(row_max_kernel, out_idx=[1], target="cuda")


def build_row_sum_exp_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def row_sum_exp_kernel(
        A: T.Tensor((M, N), dtype),
        max_row: T.Tensor((M,), dtype),
        B: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            sum_shared = T.alloc_shared((block_M,), "float32")
            T.clear(sum_shared)
            for k in T.Pipelined(T.ceildiv(N, block_N), num_stages=1):
                A_shared = T.alloc_shared((block_M, block_N), dtype)
                T.copy(A[bx * block_M:, k * block_N:], A_shared)
                for i in T.Parallel(block_M):
                    m = bx * block_M + i
                    local_sum = T.reduce(T.add, T.exp(A_shared[i, :] - max_row[m]), axis=0, init=T.cast(0.0, "float32"))
                    T.atomic_add(sum_shared[i], local_sum)
            for i in T.Parallel(block_M):
                B[bx * block_M + i] = sum_shared[i]
    return tilelang.compile(row_sum_exp_kernel, out_idx=[2], target="cuda")


def build_softmax_final_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def softmax_final_kernel(
        A: T.Tensor((M, N), dtype),
        max_row: T.Tensor((M,), dtype),
        sum_exp: T.Tensor((M,), "float32"),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[bx * block_M:, by * block_N:], A_shared)
            for i, j in T.Parallel(block_M, block_N):
                m = bx * block_M + i
                n = by * block_N + j
                if m < M and n < N:
                    B[m, n] = T.exp(A_shared[i, j] - max_row[m]) / sum_exp[m]
    return tilelang.compile(softmax_final_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self._kernel_cache = {}
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        self.running_mean = torch.zeros(out_features, dtype=torch.float32)
        self.running_var = torch.ones(out_features, dtype=torch.float32)
        self.gamma = nn.Parameter(torch.ones(out_features, dtype=torch.float16))
        self.beta = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        self.scale = nn.Parameter(torch.ones(scale_shape, dtype=torch.float16))

    def _get_fused_kernel(self, M, K, N, tl_dtype):
        key = ("fused", M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_gemm_bn_scale(M, K, N, self.bn_eps, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_row_max_kernel(self, M, N, tl_dtype):
        key = ("row_max", M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_row_max_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_row_sum_exp_kernel(self, M, N, tl_dtype):
        key = ("row_sum_exp", M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_row_sum_exp_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_softmax_final_kernel(self, M, N, tl_dtype):
        key = ("softmax_final", M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_softmax_final_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.to(torch.float16)
        M, K = x.shape
        N = self.out_features
        kernel = self._get_fused_kernel(M, K, N, "float16")
        x = kernel(x, self.weight, self.bias, self.running_mean, self.running_var, self.gamma, self.beta, self.scale)
        max_kernel = self._get_row_max_kernel(M, N, "float16")
        max_row = torch.empty(M, dtype=torch.float16, device=x.device)
        max_kernel(x, max_row)
        sum_exp_kernel = self._get_row_sum_exp_kernel(M, N, "float16")
        sum_exp = torch.empty(M, dtype=torch.float32, device=x.device)
        sum_exp_kernel(x, max_row, sum_exp)
        softmax_kernel = self._get_softmax_final_kernel(M, N, "float16")
        out = torch.empty_like(x)
        softmax_kernel(x, max_row, sum_exp, out)
        return out