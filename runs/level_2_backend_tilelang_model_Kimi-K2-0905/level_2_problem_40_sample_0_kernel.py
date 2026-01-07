import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_matmul_scale_residual_kernel(batch_size: int, in_features: int, out_features: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_matmul_scale_residual_kernel(
        A: T.Tensor((batch_size, in_features), dtype),
        B: T.Tensor((out_features, in_features), dtype),
        bias: T.Tensor((out_features,), dtype),
        C: T.Tensor((batch_size, out_features), dtype),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(block_N, block_N), T.ceildiv(batch_size, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M

            acc = T.alloc_fragment((block_M, block_N), dtype)
            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] = T.cast(0, dtype)

            for ko in range(T.ceildiv(in_features, block_K)):
                start_k = ko * block_K
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)

                for i, k in T.Parallel(block_M, block_K):
                    if start_m + i < batch_size and start_k + k < in_features:
                        A_shared[i, k] = A[start_m + i, start_k + k]
                    else:
                        A_shared[i, k] = T.cast(0, dtype)

                for j, k in T.Parallel(block_N, block_K):
                    if start_n + j < out_features and start_k + k < in_features:
                        B_shared[j, k] = B[start_n + j, start_k + k]
                    else:
                        B_shared[j, k] = T.cast(0, dtype)

                T.sync_shared()

                for ki in range(block_K):
                    for i, j in T.Parallel(block_M, block_N):
                        acc[i, j] += A_shared[i, ki] * B_shared[j, ki]

                T.sync_shared()

            for i, j in T.Parallel(block_M, block_N):
                if start_m + i < batch_size and start_n + j < out_features:
                    val = acc[i, j] + bias[start_n + j]
                    val_scaled = val * T.cast(scale, dtype)
                    C[start_m + i, start_n + j] = val_scaled + val

    return tilelang.compile(fused_matmul_scale_residual_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self._kernel_cache = {}

        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.bias)

    def _get_kernel(self, batch_size: int, tl_dtype: str):
        key = (batch_size, self.in_features, self.out_features, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_scale_residual_kernel(batch_size, self.in_features, self.out_features, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size = x.shape[0]

        kernel = self._get_kernel(batch_size, "float16")

        x_fp16 = x.half()
        weight_fp16 = self.weight.half()
        bias_fp16 = self.bias.half()

        output = kernel(x_fp16, weight_fp16, bias_fp16, self.scaling_factor)

        return output.float()