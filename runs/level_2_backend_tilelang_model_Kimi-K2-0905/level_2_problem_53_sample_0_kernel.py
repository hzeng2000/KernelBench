import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_gemm_scale_hardtanh_gelu_kernel(
    M: int, N: int, K: int,
    block_M: int = 64, block_N: int = 64, block_K: int = 32,
    threads: int = 256, dtype: str = "float16"
):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
        scaling_factor: T.float32,
        hardtanh_min: T.float32,
        hardtanh_max: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            start_m = by * block_M
            start_n = bx * block_N

            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[start_m, k * block_K], A_shared)
                T.copy(B[start_n, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                m = start_m + i
                n = start_n + j
                if m < M and n < N:
                    val = C_local[i, j] * scaling_factor
                    val = T.max(val, hardtanh_min)
                    val = T.min(val, hardtanh_max)
                    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    x = val
                    x_cubed = x * x * x
                    tanh_arg = 0.7978845608 * (x + 0.044715 * x_cubed)
                    tanh_val = T.tanh(tanh_arg)
                    gelu_val = 0.5 * x * (1.0 + tanh_val)
                    C[m, n] = gelu_val

    return tilelang.compile(kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self._kernel_cache = {}
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_scale_hardtanh_gelu_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        original_shape = x_c.shape[:-1]
        x_c = x_c.view(-1, x_c.size(-1))
        
        M, K = x_c.shape
        N = self.weight.shape[0]
        
        kernel = self._get_kernel(M, N, K, "float16")
        
        # Convert to fp16
        x_fp16 = x_c.half()
        weight_fp16 = self.weight.half()
        
        # Add bias after kernel execution
        output = kernel(x_fp16, weight_fp16, self.scaling_factor, self.hardtanh_min, self.hardtanh_max)
        
        # Add bias
        output = output + self.bias.half()
        
        return output.view(*original_shape, -1).to(x.dtype)