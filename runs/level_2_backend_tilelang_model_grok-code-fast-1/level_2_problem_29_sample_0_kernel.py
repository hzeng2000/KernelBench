import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_linear_mish_kernel(B: int, I: int, O: int, block_B: int = 64, block_O: int = 64, block_I: int = 64, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def linear_mish_kernel(
        X: T.Tensor((B, I), dtype),
        W: T.Tensor((O, I), dtype),
        Bias: T.Tensor((O,), dtype),
        Y: T.Tensor((B, O), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(O, block_O), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_B, block_I), dtype)
            B_shared = T.alloc_shared((block_I, block_O), dtype)
            C_local = T.alloc_fragment((block_B, block_O), dtype)
            T.clear(C_local)
            for k in range(T.ceildiv(I, block_I)):
                T.copy(X[bx * block_B : (bx + 1) * block_B, k * block_I : (k + 1) * block_I], A_shared)
                T.copy(W[by * block_O : (by + 1) * block_O, k * block_I : (k + 1) * block_I], B_shared, transpose_src=True)
                T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(block_B, block_O):
                temp = C_local[i, j] + Bias[by * block_O + j]
                softplus = T.log(1 + T.exp(temp))
                Y[bx * block_B + i, by * block_O + j] = temp * T.tanh(softplus)

    return tilelang.compile(linear_mish_kernel, out_idx=[3], target="cuda")


def build_mish_kernel(M: int, N: int, block_M: int = 128, block_N: int = 256, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def mish_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                if y < M and x < N:
                    softplus = T.log(1 + T.exp(A[y, x]))
                    B[y, x] = A[y, x] * T.tanh(softplus)

    return tilelang.compile(mish_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model with custom TileLang kernels for linear + mish and mish.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._kernel_cache_linear_mish = {}
        self._kernel_cache_mish = {}

    def _get_linear_mish_kernel(self, B: int, I: int, O: int, tl_dtype: str):
        key = (B, I, O, tl_dtype)
        if key not in self._kernel_cache_linear_mish:
            self._kernel_cache_linear_mish[key] = build_linear_mish_kernel(B, I, O, dtype=tl_dtype)
        return self._kernel_cache_linear_mish[key]

    def _get_mish_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._kernel_cache_mish:
            self._kernel_cache_mish[key] = build_mish_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache_mish[key]

    def forward(self, x):
        # Cast to half for FP16
        x = x.half()
        self.linear.weight.data = self.linear.weight.data.half()
        self.linear.bias.data = self.linear.bias.data.half()
        
        B, I = x.shape
        O = self.linear.out_features
        
        # First kernel: linear + mish
        kernel1 = self._get_linear_mish_kernel(B, I, O, "float16")
        x = kernel1(x, self.linear.weight, self.linear.bias)
        
        # Second kernel: mish
        kernel2 = self._get_mish_kernel(B, O, "float16")
        x = kernel2(x)
        
        return x