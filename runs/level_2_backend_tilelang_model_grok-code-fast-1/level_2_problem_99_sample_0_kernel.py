import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_gelu_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_linear_gelu_kernel(
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
                T.copy(A[by * block_M : (by+1)*block_M, k*block_K : (k+1)*block_K], A_shared)
                T.copy(B[k*block_K : (k+1)*block_K, bx * block_N : (bx+1)*block_N], B_shared)
                with T.Block():
                    for i, j, kk in T.grid(block_M, block_N, block_K):
                        with T.block():
                            C_local[i, j] += A_shared[i, kk] * B_shared[kk, j]
            with T.Block():
                for i, j in T.grid(block_M, block_N):
                    y = by * block_M + i
                    x = bx * block_N + j
                    if y < M and x < N:
                        temp = C_local[i, j] + bias[x]
                        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        sqrt_2_pi = 0.7978845608028654
                        coeff = 0.044715
                        tanh_arg = sqrt_2_pi * (temp + coeff * temp * temp * temp)
                        gelu = 0.5 * temp * (1 + T.tanh(tanh_arg))
                        C[y, x] = gelu

    return tilelang.compile(fused_linear_gelu_kernel, out_idx=[3], target="cuda")


def build_softmax_kernel(M: int, N: int, dtype: str = "float16"):
    
    @T.prim_func
    def softmax_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, 1, threads=1024) as (bx,):
            max_val = T.reduce_max(A[bx, :], axis=1, init=-float('inf'))
            sum_val = T.reduce_sum(T.exp(A[bx, :] - max_val), axis=1, init=0)
            for i in T.Parallel(N):
                B[bx, i] = T.exp(A[bx, i] - max_val) / sum_val

    return tilelang.compile(softmax_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model with fused linear+GELU and custom softmax using TileLang kernels.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self._kernel_cache = {}

    def _get_fused_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = ("fused", M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_gelu_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_softmax_kernel(self, M: int, N: int, tl_dtype: str):
        key = ("softmax", M, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_softmax_kernel(M, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        # Convert to half precision for FP16 optimization
        x = x.half()
        weight = self.weight.half()
        bias = self.bias.half()
        
        M, K = x.shape
        N = self.weight.shape[0]
        
        # Fused linear + GELU
        fused_kernel = self._get_fused_kernel(M, K, N, "float16")
        x = fused_kernel(x, weight.t(), bias)
        
        # Softmax
        softmax_kernel = self._get_softmax_kernel(M, N, "float16")
        output = softmax_kernel(x)
        
        return output.float()  # Convert back to float32 for compatibility