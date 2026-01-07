import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def gelu(x):
    # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = math.sqrt(2 / math.pi)
    return 0.5 * x * (1.0 + T.tanh(sqrt_2_pi * (x + 0.044715 * x * x * x)))


def build_fused_kernel(M: int, K: int, N: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),
        divisor: T.Tensor((), dtype),
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
            
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = by * block_M + local_y
                x = bx * block_N + local_x
                if y < M and x < N:
                    val = C_local[local_y, local_x] + bias[x]
                    val = val / divisor[()]
                    val = gelu(val)
                    C[y, x] = val
    
    return tilelang.compile(fused_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size, dtype=torch.float16).cuda())
        self.bias = nn.Parameter(torch.randn(output_size, dtype=torch.float16).cuda())
        self.divisor = divisor
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half().contiguous()
        original_shape = x.shape
        x = x.view(-1, x.size(-1))
        M, K = x.shape
        N = self.weight.shape[0]
        kernel = self._get_kernel(M, K, N, "float16")
        divisor_tensor = torch.tensor(self.divisor, dtype=torch.float16).cuda()
        C = kernel(x, self.weight.t(), self.bias, divisor_tensor)
        return C.view(original_shape)