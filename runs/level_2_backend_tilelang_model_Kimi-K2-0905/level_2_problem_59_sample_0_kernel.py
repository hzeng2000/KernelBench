import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math

def build_fused_matmul_swish_scale_kernel(M: int, N: int, K: int, 
                                          block_M: int = 64, block_N: int = 64, block_K: int = 32,
                                          threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            start_m = by * block_M
            start_n = bx * block_N
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                start_k = k * block_K
                
                # Load A tile to shared memory
                for i, j in T.Parallel(block_M, block_K):
                    global_m = start_m + i
                    global_k = start_k + j
                    if global_m < M and global_k < K:
                        A_shared[i, j] = A[global_m, global_k]
                    else:
                        A_shared[i, j] = T.cast(0, dtype)
                
                # Load B tile to shared memory
                for i, j in T.Parallel(block_K, block_N):
                    global_k = start_k + i
                    global_n = start_n + j
                    if global_k < K and global_n < N:
                        B_shared[i, j] = B[global_k, global_n]
                    else:
                        B_shared[i, j] = T.cast(0, dtype)
                
                T.gemm(A_shared, B_shared, C_local, trans_B=True)
            
            # Apply Swish activation and scaling
            for i, j in T.Parallel(block_M, block_N):
                global_m = start_m + i
                global_n = start_n + j
                if global_m < M and global_n < N:
                    val = C_local[i, j]
                    sigmoid = T.cast(1.0, dtype) / (T.cast(1.0, dtype) + T.exp(-val))
                    swish = val * sigmoid
                    C[global_m, global_n] = swish * T.cast(scale, dtype)
    
    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, dtype: str):
        key = (M, N, K, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_swish_scale_kernel(M, N, K, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        original_shape = x_c.shape[:-1]
        M = x_c.numel() // self.in_features
        K = self.in_features
        N = self.out_features
        
        x_2d = x_c.view(M, K)
        
        # Convert to fp16 for kernel computation
        x_fp16 = x_2d.half()
        weight_fp16 = self.weight.t().half()
        
        kernel = self._get_kernel(M, N, K, "float16")
        output_fp16 = kernel(x_fp16, weight_fp16, self.scaling_factor)
        
        # Add bias
        output_fp16 = output_fp16 + self.bias.half()
        
        # Convert back to fp32 and reshape
        output = output_fp16.float()
        return output.view(*original_shape, N)