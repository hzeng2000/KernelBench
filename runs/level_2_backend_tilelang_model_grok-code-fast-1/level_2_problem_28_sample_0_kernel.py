import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_linear_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, block_K: int = 8, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def linear_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.use_swizzle(10)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[bx * block_M: bx * block_M + block_M, k * block_K: k * block_K + block_K], A_shared)
                T.copy(B[by * block_N: by * block_N + block_N, k * block_K: k * block_K + block_K], B_shared)
                with T.block(""):
                    for i, j, kk in T.grid(block_M, block_N, block_K):
                        with T.block("update"):
                            C_local[i, j] += A_shared[i, kk] * B_shared[j, kk]
            T.copy(C_local, C[bx * block_M: bx * block_M + block_M, by * block_N: by * block_N + block_N])
            with T.block(""):
                for i, j in T.grid(block_M, block_N):
                    if bx * block_M + i < M and by * block_N + j < N:
                        C[bx * block_M + i, by * block_N + j] += bias[j]
    return tilelang.compile(linear_kernel, out_idx=[3], target="cuda")


def build_instance_norm_kernel(M: int, N: int, dtype: str = "float16"):
    @T.prim_func
    def instance_norm_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        eps: T.float32,
    ):
        for i in T.Parallel(M):
            mean = T.reduce(T.sum, A[i, :], axis=0) / N
            var_sum = T.reduce(T.sum, (A[i, :] - mean) ** 2, axis=0) / N
            for j in T.Parallel(N):
                B[i, j] = (A[i, j] - mean) / T.sqrt(var_sum + eps)
    return tilelang.compile(instance_norm_kernel, out_idx=[1], target="cuda")


def build_add_mul_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def add_mul_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                if y < M and x < N:
                    C[y, x] = (A[y, x] + B[y, x]) * B[y, x]
    return tilelang.compile(add_mul_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self._kernel_cache = {}

    def _get_linear_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = ("linear", M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_linear_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_instance_norm_kernel(self, M: int, N: int, tl_dtype: str):
        key = ("instance_norm", M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_instance_norm_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_add_mul_kernel(self, M: int, N: int, tl_dtype: str):
        key = ("add_mul", M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_add_mul_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x, y):
        x = x.half()
        y = y.half()
        weight = self.bmm.weight.half()
        bias = self.bmm.bias.half()
        batch, in_f = x.shape
        out_f = weight.shape[0]
        eps = self.instance_norm.eps
        kernel_linear = self._get_linear_kernel(batch, in_f, out_f, "float16")
        temp = kernel_linear(x, weight, bias)
        kernel_norm = self._get_instance_norm_kernel(batch, out_f, "float16")
        temp2 = kernel_norm(temp, eps)
        kernel_add_mul = self._get_add_mul_kernel(batch, out_f, "float16")
        result = kernel_add_mul(temp2, y)
        return result.float()