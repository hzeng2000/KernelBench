import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_matmul_kernel(M: int, K: int, N: int, block_M: int = 64, block_N: int = 64, block_K: int = 64, threads: int = 128, dtype: str = "float16"):
    
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
                T.copy(A[by * block_M : (by+1)*block_M, k*block_K : (k+1)*block_K], A_shared)
                T.copy(B[k*block_K : (k+1)*block_K, bx * block_N : (bx+1)*block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            # Add bias to C_local
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += bias[bx * block_N + j]
            
            T.copy(C_local, C[by * block_M : (by+1)*block_M, bx * block_N : (bx+1)*block_N])

    return tilelang.compile(matmul_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication (Gemm) with custom TileLang kernel, followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        else:
            self.bias_param = None
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_matmul_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        # Convert to FP16
        x = x.to(torch.float16)
        # Gemm with custom kernel
        M, K = x.shape
        N = self.out_features
        kernel = self._get_kernel(M, K, N, "float16")
        x = kernel(x, self.weight, self.bias_param)
        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)
        # LeakyReLU
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # LeakyReLU
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # GELU
        x = torch.nn.functional.gelu(x)
        # GELU
        x = torch.nn.functional.gelu(x)
        return x