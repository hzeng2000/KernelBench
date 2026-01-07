import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_matmul_bias_kernel(M: int, K: int, N: int, block_M: int = 64, block_K: int = 64, block_N: int = 64, threads: int = 256, dtype: str = "float16"):
    @T.prim_func
    def matmul_bias_kernel(
        A: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M: (by + 1) * block_M, k * block_K: (k + 1) * block_K], A_shared)
                T.copy(W[bx * block_N: (bx + 1) * block_N, k * block_K: (k + 1) * block_K], W_shared)
                T.gemm(A_shared, W_shared, C_local)
            T.copy(C_local, C[by * block_M: (by + 1) * block_M, bx * block_N: (bx + 1) * block_N])
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] += bias[bx * block_N + j]
    return tilelang.compile(matmul_bias_kernel, out_idx=[3], target="cuda")


def build_elem_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def elem_kernel(
        A: T.Tensor((M, N), dtype),
        multiplier: T.Tensor((), dtype),
        negative_slope: T.Tensor((), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = by * block_M + local_y
                x = bx * block_N + local_x
                temp = A[y, x] * multiplier[()]
                C[y, x] = T.select(temp > 0, temp, temp * negative_slope[()])
    return tilelang.compile(elem_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.in_features = in_features
        self.out_features = out_features
        self._kernel_cache_matmul = {}
        self._kernel_cache_elem = {}

    def _get_matmul_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache_matmul:
            self._kernel_cache_matmul[key] = build_matmul_bias_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache_matmul[key]

    def _get_elem_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache_elem:
            self._kernel_cache_elem[key] = build_elem_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache_elem[key]

    def forward(self, x):
        x_c = x.contiguous()
        original_shape = x_c.shape
        x_c = x_c.view(-1, self.in_features)
        M, K = x_c.shape
        N = self.out_features
        kernel_matmul = self._get_matmul_kernel(M, K, N, "float16")
        temp = kernel_matmul(x_c.to(torch.float16), self.weight, self.bias)
        kernel_elem = self._get_elem_kernel(M, N, "float16")
        multiplier_tensor = torch.tensor(self.multiplier, dtype=torch.float16, device=x.device)
        negative_slope_tensor = torch.tensor(self.negative_slope, dtype=torch.float16, device=x.device)
        out = kernel_elem(temp, multiplier_tensor, negative_slope_tensor)
        return out.view(original_shape)