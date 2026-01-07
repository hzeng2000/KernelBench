import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_matmul_kernel(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[k * block_K : (k + 1) * block_K, bx * block_N : (bx + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])
            for i, j in T.Parallel(block_M, block_N):
                y = by * block_M + i
                x = bx * block_N + j
                if y < M and x < N:
                    C[y, x] += bias[x]
    return tilelang.compile(matmul_kernel, out_idx=[3], target="cuda")


def build_leaky_relu_scale_kernel(M: int, N: int, negative_slope: float, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def leaky_relu_scale_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = by * block_M + local_y
                x = bx * block_N + local_x
                if y < M and x < N:
                    val = A[y, x]
                    B[y, x] = T.select(val > 0, 2 * val, 2 * negative_slope * val)
    return tilelang.compile(leaky_relu_scale_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.negative_slope = negative_slope
        self._kernel_cache = {}

    def _get_matmul_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = ("matmul", M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_matmul_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_leaky_kernel(self, M: int, N: int, negative_slope: float, tl_dtype: str):
        key = ("leaky", M, N, negative_slope, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_leaky_relu_scale_kernel(M, N, negative_slope, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = x.half()
        M, K = x.shape
        N = self.weight.shape[0]
        kernel = self._get_matmul_kernel(M, N, K, "float16")
        x = kernel(x, self.weight.t(), self.bias)
        x = self.gn(x)
        M, N = x.shape
        kernel2 = self._get_leaky_kernel(M, N, self.negative_slope, "float16")
        x = kernel2(x)
        return x.float()