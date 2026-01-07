import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_matmul_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 256, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc((block_M, block_K), dtype)
            B_shared = T.alloc((block_K, block_N), dtype)
            C_local = T.alloc((block_M, block_N), "float32")
            T.clear(C_local)
            for k in range(T.ceildiv(K, block_K)):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K].T, B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local + bias[by * block_N : (by + 1) * block_N], C[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return tilelang.compile(matmul_kernel, out_idx=[3], target="cuda")


def build_final_kernel(M: int, N: int, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def final_kernel(
        X: T.Tensor((M, N), dtype),
        mean_val: T.Tensor((M, 1), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, 1), T.ceildiv(N, block_N), threads=threads) as (by, bx):
            start_y = by * 1
            start_x = bx * block_N
            for local_y, local_x in T.Parallel(1, block_N):
                y = start_y + local_y
                x = start_x + local_x
                Y[y, x] = T.gelu(X[y, x] - mean_val[y, 0])

    return tilelang.compile(final_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_dim = max_dim
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self._matmul_kernel_cache = {}
        self._final_kernel_cache = {}

    def _get_matmul_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._matmul_kernel_cache:
            self._matmul_kernel_cache[key] = build_matmul_kernel(M, K, N, dtype=tl_dtype)
        return self._matmul_kernel_cache[key]

    def _get_final_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._final_kernel_cache:
            self._final_kernel_cache[key] = build_final_kernel(M, N, dtype=tl_dtype)
        return self._final_kernel_cache[key]

    def forward(self, x):
        x = x.to(torch.float16)
        weight = self.weight.to(torch.float16)
        bias = self.bias.to(torch.float16)
        
        batch_size, in_f = x.shape
        out_f = self.out_features
        
        matmul_kernel = self._get_matmul_kernel(batch_size, in_f, out_f, "float16")
        x = matmul_kernel(x, weight, bias)
        
        max_val = torch.max(x, dim=self.max_dim, keepdim=True).values
        x = x - max_val
        mean_val = x.mean(dim=1, keepdim=True)
        
        final_kernel = self._get_final_kernel(batch_size, out_f, "float16")
        x = final_kernel(x, mean_val)
        
        return x.to(torch.float32)