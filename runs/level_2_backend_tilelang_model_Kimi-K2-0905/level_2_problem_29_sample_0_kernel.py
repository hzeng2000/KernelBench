import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_mish_mish_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_linear_mish_mish_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M

            # Allocate shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            # Initialize C_local to zero
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.cast(0.0, dtype)

            # Loop over K dimension
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                k_start = k * block_K

                # Load A into shared memory
                for i, j in T.Parallel(block_M, block_K):
                    global_i = start_m + i
                    global_j = k_start + j
                    if global_i < M and global_j < K:
                        A_shared[i, j] = A[global_i, global_j]
                    else:
                        A_shared[i, j] = T.cast(0.0, dtype)

                # Load B into shared memory (transposed)
                for i, j in T.Parallel(block_N, block_K):
                    global_i = start_n + i
                    global_j = k_start + j
                    if global_i < N and global_j < K:
                        B_shared[i, j] = B[global_i, global_j]
                    else:
                        B_shared[i, j] = T.cast(0.0, dtype)

                # Compute partial matmul
                for i, j in T.Parallel(block_M, block_N):
                    for kk in range(block_K):
                        C_local[i, j] += A_shared[i, kk] * B_shared[j, kk]

            # Apply Mish activation twice
            for i, j in T.Parallel(block_M, block_N):
                global_i = start_m + i
                global_n = start_n + j
                if global_i < M and global_n < N:
                    # First Mish
                    x = C_local[i, j]
                    tanh_arg = T.tanh(T.log1p(T.exp(x)))
                    mish1 = x * tanh_arg
                    # Second Mish
                    tanh_arg2 = T.tanh(T.log1p(T.exp(mish1)))
                    mish2 = mish1 * tanh_arg2
                    C[global_i, global_n] = mish2

    return tilelang.compile(fused_linear_mish_mish_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self._kernel_cache = {}
        # Initialize weight
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_mish_mish_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        M, K = x_c.shape
        N = self.out_features
        
        # Convert to fp16
        x_fp16 = x_c.half()
        weight_fp16 = self.weight.half()
        
        kernel = self._get_kernel(M, K, N, "float16")
        output = kernel(x_fp16, weight_fp16)
        
        return output