import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_matmul_kernel(M, K, N, block_M=128, block_N=256, block_K=32, threads=128, dtype="float16"):
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M:, k * block_K:], A_shared)
                T.copy(B[k * block_K:, bx * block_N:], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M:, bx * block_N:])

    return tilelang.compile(matmul_kernel, out_idx=[2], target="cuda")


def build_avgpool_kernel(batch_size, out_features, pooled_features, pool_size, block_B=1, block_P=128, threads=128, dtype="float16"):
    @T.prim_func
    def avgpool_kernel(
        inp: T.Tensor((batch_size, out_features), dtype),
        out: T.Tensor((batch_size, pooled_features), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_B), T.ceildiv(pooled_features, block_P), threads=threads) as (bx, by):
            b = bx * block_B
            start_p = by * block_P
            for local_p in T.Parallel(block_P):
                p = start_p + local_p
                if p < pooled_features:
                    sum_val = T.alloc_fragment((1,), dtype)
                    sum_val[0] = 0.0
                    for k in T.serial(pool_size):
                        sum_val[0] += inp[b, p * pool_size + k]
                    out[b, p] = sum_val[0] / pool_size

    return tilelang.compile(avgpool_kernel, out_idx=[1], target="cuda")


def build_post_kernel(batch_size, pooled_features, scale_factor, block_B=1, block_P=512, threads=512, dtype="float16"):
    @T.prim_func
    def post_kernel(
        inp: T.Tensor((batch_size, pooled_features), dtype),
        out: T.Tensor((batch_size,), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_B), 1, threads=threads) as (bx):
            b = bx * block_B
            shared = T.alloc_shared((block_P,), dtype)
            for local_p in T.Parallel(block_P):
                p = local_p
                x = inp[b, p] * scale_factor
                x_cub = x * x * x
                tanh_arg = T.sqrt(2 / 3.141592653589793) * (x + 0.044715 * x_cub)
                gelu_val = 0.5 * x * (1 + T.tanh(tanh_arg))
                shared[p] = gelu_val
            T.sync_thread()
            num_steps = int(math.log2(block_P))
            for i in T.serial(num_steps):
                stride = 1 << (num_steps - 1 - i)
                if T.get_thread_id() < stride:
                    shared[T.get_thread_id()] = T.maximum(shared[T.get_thread_id()], shared[T.get_thread_id() + stride])
                T.sync_thread()
            out[b] = shared[0]

    return tilelang.compile(post_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.pooled_features = out_features // pool_kernel_size
        self._kernel_cache = {}

    def _get_matmul_kernel(self, M, K, N, dtype):
        key = ("matmul", M, K, N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_matmul_kernel(M, K, N, dtype=dtype)
        return self._kernel_cache[key]

    def _get_avgpool_kernel(self, batch_size, out_features, pooled_features, pool_size, dtype):
        key = ("avgpool", batch_size, out_features, pooled_features, pool_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_avgpool_kernel(batch_size, out_features, pooled_features, pool_size, dtype=dtype)
        return self._kernel_cache[key]

    def _get_post_kernel(self, batch_size, pooled_features, scale_factor, dtype):
        key = ("post", batch_size, pooled_features, scale_factor, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_post_kernel(batch_size, pooled_features, scale_factor, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half()
        batch_size, in_features = x.shape
        out_features = self.matmul.out_features
        pooled_features = self.pooled_features
        pool_size = self.pool_kernel_size
        scale_factor = self.scale_factor

        # Matmul: x @ weight.T
        weight = self.matmul.weight.half()
        matmul_kernel = self._get_matmul_kernel(batch_size, in_features, out_features, "float16")
        matmul_out = torch.empty(batch_size, out_features, dtype=torch.float16, device=x.device)
        matmul_kernel(x, weight.t(), matmul_out)

        # AvgPool
        avgpool_kernel = self._get_avgpool_kernel(batch_size, out_features, pooled_features, pool_size, "float16")
        pooled_out = torch.empty(batch_size, pooled_features, dtype=torch.float16, device=x.device)
        avgpool_kernel(matmul_out, pooled_out)

        # GELU, scale, max
        post_kernel = self._get_post_kernel(batch_size, pooled_features, scale_factor, "float16")
        final_out = torch.empty(batch_size, dtype=torch.float16, device=x.device)
        post_kernel(pooled_out, final_out)

        return final_out.float()