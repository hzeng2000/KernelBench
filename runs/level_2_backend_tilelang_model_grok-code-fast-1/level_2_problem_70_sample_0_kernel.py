import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_gemm_sigmoid_scale_residual_kernel(
    batch: int,
    input_size: int,
    hidden_size: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    threads: int = 128,
    dtype: str = "float16"
):
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((batch, input_size), dtype),
        W: T.Tensor((hidden_size, input_size), dtype),
        B: T.Tensor((hidden_size,), dtype),
        scaling_factor: T.Tensor((), dtype),
        C: T.Tensor((batch, hidden_size), dtype),
    ):
        with T.Kernel(T.ceildiv(hidden_size, block_N), T.ceildiv(batch, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(input_size, block_K), num_stages=3):
                T.copy(A[by * block_M: (by + 1) * block_M, k * block_K: (k + 1) * block_K], A_shared)
                T.copy(W[k * block_K: (k + 1) * block_K, bx * block_N: (bx + 1) * block_N], W_shared)
                for i, j in T.Parallel(block_M, block_N):
                    for kk in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, kk] * W_shared[kk, j]
            for i, j in T.Parallel(block_M, block_N):
                gemm_out = C_local[i, j] + B[bx * block_N + j]
                sigmoid_out = 1.0 / (1.0 + T.exp(-gemm_out))
                scaled = sigmoid_out * scaling_factor[()]
                C[by * block_M + i, bx * block_N + j] = scaled + gemm_out

    return tilelang.compile(fused_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(hidden_size, dtype=torch.float16))
        self.scaling_factor = nn.Parameter(torch.tensor(scaling_factor, dtype=torch.float16))
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, input_size: int, hidden_size: int, tl_dtype: str):
        key = (batch, input_size, hidden_size, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_gemm_sigmoid_scale_residual_kernel(
                batch, input_size, hidden_size, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        batch, input_size = x.shape
        hidden_size = self.weight.shape[0]
        kernel = self._get_kernel(batch, input_size, hidden_size, "float16")
        C = kernel(x, self.weight, self.bias, self.scaling_factor)
        return C