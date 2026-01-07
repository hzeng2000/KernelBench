import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
from tilelang import ReduceOp


def build_fused_matmul_kernel(batch_size: int, input_size: int, hidden_size: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def fused_matmul_kernel(
        A: T.Tensor((batch_size, input_size), dtype),
        B: T.Tensor((input_size, hidden_size), dtype),
        bias: T.Tensor((hidden_size,), dtype),
        scale_factor: T.float32,
        clamp_min: T.float32,
        clamp_max: T.float32,
        C: T.Tensor((batch_size, hidden_size), dtype),
    ):
        with T.Kernel(T.ceildiv(hidden_size, block_N), T.ceildiv(batch_size, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            T.clear(C_local)
            for k in T.serial(T.ceildiv(input_size, block_K)):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[k * block_K : (k + 1) * block_K, bx * block_N : (bx + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

            # Fuse post-ops: add bias, * scale_factor, * 2, clamp
            for i, j in T.Parallel(block_M, block_N):
                y = by * block_M + i
                x = bx * block_N + j
                if y < batch_size and x < hidden_size:
                    val = C[y, x] + bias[x]
                    val = val * scale_factor * 2.0
                    val = T.clamp(val, clamp_min, clamp_max)
                    C[y, x] = val

    return tilelang.compile(fused_matmul_kernel, out_idx=[6], target="cuda")


def build_logsumexp_kernel(batch_size: int, hidden_size: int, block_B: int = 32, block_H: int = 1024, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def logsumexp_kernel(
        A: T.Tensor((batch_size, hidden_size), dtype),
        C: T.Tensor((batch_size, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_B), 1, threads=threads) as bx:
            A_shared = T.alloc_shared((block_B, block_H), dtype)
            max_local = T.alloc_fragment((block_B,), dtype)
            sum_local = T.alloc_fragment((block_B,), dtype)

            T.fill(max_local, T.min_value(dtype))
            T.fill(sum_local, 0.0)

            for h in T.serial(T.ceildiv(hidden_size, block_H)):
                T.copy(A[bx * block_B : (bx + 1) * block_B, h * block_H : (h + 1) * block_H], A_shared)
                for i in T.Parallel(block_B):
                    for j in T.serial(block_H):
                        val = A_shared[i, j]
                        max_local[i] = T.max(max_local[i], val)
                        sum_local[i] = sum_local[i] + T.exp(val - max_local[i])

            for i in T.Parallel(block_B):
                logsum = max_local[i] + T.log(sum_local[i])
                C[bx * block_B + i, 0] = logsum

    return tilelang.compile(logsumexp_kernel, out_idx=[1], target="cuda")


def build_final_kernel(batch_size: int, threads: int = 128, dtype: str = "float16"):
    @T.prim_func
    def final_kernel(
        A: T.Tensor((batch_size, 1), dtype),
        C: T.Tensor((batch_size, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, threads), 1, threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                if idx < batch_size:
                    z = A[idx, 0]
                    # mish(z) = z * tanh(softplus(z))
                    softplus = T.log(1.0 + T.exp(z))
                    tanh_softplus = T.tanh(softplus)
                    mish_z = z * tanh_softplus
                    C[idx, 0] = z * mish_z

    return tilelang.compile(final_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self._kernel_cache = {}

    def _get_fused_kernel(self, batch_size: int, input_size: int, hidden_size: int, tl_dtype: str):
        key = ("fused", batch_size, input_size, hidden_size, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_kernel(batch_size, input_size, hidden_size, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_logsumexp_kernel(self, batch_size: int, hidden_size: int, tl_dtype: str):
        key = ("logsumexp", batch_size, hidden_size, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_logsumexp_kernel(batch_size, hidden_size, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_final_kernel(self, batch_size: int, tl_dtype: str):
        key = ("final", batch_size, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_final_kernel(batch_size, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half().contiguous()
        batch_size, input_size = x.shape
        hidden_size = self.matmul.out_features

        W = self.matmul.weight.data.t().half().contiguous()  # (input_size, hidden_size)
        b = self.matmul.bias.data.half().contiguous()  # (hidden_size,)

        fused_kernel = self._get_fused_kernel(batch_size, input_size, hidden_size, "float16")
        y = fused_kernel(x, W, b, self.scale_factor, self.clamp_min, self.clamp_max)

        logsumexp_kernel = self._get_logsumexp_kernel(batch_size, hidden_size, "float16")
        z = logsumexp_kernel(y)

        final_kernel = self._get_final_kernel(batch_size, "float16")
        output = final_kernel(z)

        return output.float()