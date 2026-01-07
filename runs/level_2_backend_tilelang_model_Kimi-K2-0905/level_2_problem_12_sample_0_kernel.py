import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemm_mul_leakyrelu_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_mul_leakyrelu_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
        multiplier: T.float32,
        negative_slope: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_m = by * block_M
            start_n = bx * block_N

            # Allocate shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            
            # Allocate local accumulator
            C_local = T.alloc_fragment((block_M, block_N), "float32", scope="local")

            # Initialize accumulator
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.float32(0.0)

            # Main computation loop
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                # Load A tile to shared memory
                for i, j in T.Parallel(block_M, block_K):
                    global_i = start_m + i
                    global_j = k * block_K + j
                    if global_i < M and global_j < K:
                        A_shared[i, j] = A[global_i, global_j]
                    else:
                        A_shared[i, j] = T.float16(0.0)

                # Load B tile to shared memory (transposed)
                for i, j in T.Parallel(block_N, block_K):
                    global_i = start_n + i
                    global_j = k * block_K + j
                    if global_i < N and global_j < K:
                        B_shared[i, j] = B[global_i, global_j]
                    else:
                        B_shared[i, j] = T.float16(0.0)

                # Compute GEMM tile
                for i, j in T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        C_local[i, j] += T.cast(A_shared[i, kk], "float32") * T.cast(B_shared[j, kk], "float32")

            # Apply multiplier and LeakyReLU, then store result
            for i, j in T.Parallel(block_M, block_N):
                global_i = start_m + i
                global_j = start_n + j
                if global_i < M and global_j < N:
                    val = C_local[i, j] * multiplier
                    # LeakyReLU: max(val, 0) + negative_slope * min(val, 0)
                    C[global_i, global_j] = T.cast(
                        T.max(val, T.float32(0.0)) + negative_slope * T.min(val, T.float32(0.0)), 
                        dtype
                    )

    return tilelang.compile(gemm_mul_leakyrelu_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self._kernel_cache = {}
        # Initialize weight
        nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain('leaky_relu', negative_slope))

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_mul_leakyrelu_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        M, K = x_c.shape
        N = self.out_features
        
        # Convert to fp16
        x_fp16 = x_c.half()
        weight_fp16 = self.weight.half()
        
        kernel = self._get_kernel(M, N, K, "float16")
        out = kernel(x_fp16, weight_fp16, self.multiplier, self.negative_slope)
        
        return out