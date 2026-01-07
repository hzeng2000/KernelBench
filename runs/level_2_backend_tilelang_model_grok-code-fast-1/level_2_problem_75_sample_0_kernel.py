import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_matmul_kernel(M: int, K: int, N: int, dtype: str = "float16"):
    block_M = 64
    block_N = 64
    block_K = 32
    threads = 128

    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                for i, j in T.Parallel(block_M, block_N):
                    for l in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, l] * B_shared[j, l]
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_local[i, j] + bias[bx * block_N + j]

    return tilelang.compile(matmul_kernel, out_idx=[3], target="cuda")


def build_min_kernel(M: int, N: int, dtype: str = "float16"):
    block_M = 128
    block_N = 256
    threads = 128

    @T.prim_func
    def min_kernel(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M,), dtype)
            T.fill(C_local, T.max_value(dtype))
            for k in T.Pipelined(T.ceildiv(N, block_N), num_stages=3):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_N : (k + 1) * block_N], A_shared)
                for i in T.Parallel(block_M):
                    for j in T.Serial(block_N):
                        C_local[i] = T.min(C_local[i], A_shared[i, j])
            for i in T.Parallel(block_M):
                C[bx * block_M + i, 0] = C_local[i]

    return tilelang.compile(min_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias_linear = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float16))
        self.matmul_kernel = build_matmul_kernel(1024, in_features, out_features)
        self.min_kernel = build_min_kernel(1024, out_features)

    def forward(self, x):
        x = x.contiguous().to(torch.float16)
        x = self.matmul_kernel(x, self.weight, self.bias_linear)
        x = self.group_norm(x)
        x = self.min_kernel(x)
        x = x + self.bias
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]