import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_gemm_swish_clamp_tanh_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_n = bx * block_N
            start_m = by * block_M

            # Allocate shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype, accum=True)

            # Initialize accumulator
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.cast(0, dtype)

            # Main computation loop
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                # Load A tile to shared memory
                for i, j in T.Parallel(block_M, block_K):
                    global_i = start_m + i
                    global_j = k * block_K + j
                    if global_i < M and global_j < K:
                        A_shared[i, j] = A[global_i, global_j]
                    else:
                        A_shared[i, j] = T.cast(0, dtype)

                # Load B tile to shared memory (transposed)
                for i, j in T.Parallel(block_N, block_K):
                    global_i = start_n + i
                    global_j = k * block_K + j
                    if global_i < N and global_j < K:
                        B_shared[i, j] = B[global_i, global_j]
                    else:
                        B_shared[i, j] = T.cast(0, dtype)

                # Compute GEMM
                for i, j in T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        C_local[i, j] += A_shared[i, kk] * B_shared[j, kk]

            # Apply bias, swish, divide, clamp, tanh, clamp
            for i, j in T.Parallel(block_M, block_N):
                global_i = start_m + i
                global_n = start_n + j
                
                if global_i < M and global_n < N:
                    # Add bias
                    val = C_local[i, j] + bias[j]
                    
                    # Swish activation: x * sigmoid(x)
                    sigmoid = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-val))
                    val = val * sigmoid
                    
                    # Divide by 2.0
                    val = val / T.cast(2.0, dtype)
                    
                    # Clamp between -1.0 and 1.0
                    val = T.max(val, T.cast(-1.0, dtype))
                    val = T.min(val, T.cast(1.0, dtype))
                    
                    # Tanh activation
                    val = T.tanh(val)
                    
                    # Final clamp between -1.0 and 1.0
                    val = T.max(val, T.cast(-1.0, dtype))
                    val = T.min(val, T.cast(1.0, dtype))
                    
                    C[global_i, global_n] = val

    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._kernel_cache = {}

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_gemm_swish_clamp_tanh_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        M, K = x_c.shape
        N = self.out_features
        
        # Convert to fp16
        x_fp16 = x_c.half()
        weight_fp16 = self.weight.half()
        bias_fp16 = self.bias.half() if self.bias is not None else torch.zeros(N, dtype=torch.float16, device=x.device)
        
        kernel = self._get_kernel(M, N, K, "float16")
        output = kernel(x_fp16, weight_fp16, bias_fp16)
        
        return output