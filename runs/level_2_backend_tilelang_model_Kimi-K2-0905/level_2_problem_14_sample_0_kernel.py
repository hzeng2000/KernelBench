import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_gemm_div_sum_scale_kernel(batch_size: int, input_size: int, hidden_size: int, scaling_factor: float,
                                          block_M: int = 64, block_N: int = 64, block_K: int = 32,
                                          threads: int = 256, dtype: str = "float16"):
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((batch_size, input_size), dtype),
        B: T.Tensor((hidden_size, input_size), dtype),
        C: T.Tensor((batch_size, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_M), threads=threads) as bx:
            start_m = bx * block_M
            local_sum = T.alloc_fragment((block_M,), dtype)

            for local_i in T.Parallel(block_M):
                local_sum[local_i] = T.cast(0.0, dtype)

            for k in T.Pipelined(T.ceildiv(input_size, block_K), num_stages=2):
                for n in T.Pipelined(T.ceildiv(hidden_size, block_N), num_stages=2):
                    A_frag = T.alloc_fragment((block_M, block_K), dtype)
                    B_frag = T.alloc_fragment((block_N, block_K), dtype)
                    C_frag = T.alloc_fragment((block_M, block_N), dtype)

                    T.copy(A[start_m + T.thread_binding(0, block_M), k * block_K + T.thread_binding(0, block_K)], A_frag)
                    T.copy(B[n * block_N + T.thread_binding(0, block_N), k * block_K + T.thread_binding(0, block_K)], B_frag)

                    T.gemm(A_frag, B_frag, C_frag, trans_B=True)

                    for local_i, local_j in T.Parallel(block_M, block_N):
                        local_sum[local_i] += C_frag[local_i, local_j] * T.cast(0.5 * scaling_factor, dtype)

            for local_i in T.Parallel(block_M):
                if start_m + local_i < batch_size:
                    C[start_m + local_i, 0] = local_sum[local_i]

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, input_size: int, hidden_size: int, scaling_factor: float):
        key = (batch_size, input_size, hidden_size, scaling_factor)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_gemm_div_sum_scale_kernel(
                batch_size, input_size, hidden_size, scaling_factor, dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        batch_size, input_size = x_c.shape
        hidden_size = self.weight.shape[0]

        kernel = self._get_kernel(batch_size, input_size, hidden_size, self.scaling_factor)
        output = kernel(x_c.half(), self.weight.half())

        return output