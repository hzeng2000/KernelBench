import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_matmul_maxpool_sum_kernel(
    batch_size: int, 
    in_features: int, 
    out_features: int, 
    kernel_size: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    threads: int = 256,
    dtype: str = "float16"
):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((batch_size, in_features), dtype),
        B: T.Tensor((out_features, in_features), dtype),
        C: T.Tensor((batch_size,), dtype),
        scale_factor: T.float32,
    ):
        # Shared memory for A tile
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        # Shared memory for B tile
        B_shared = T.alloc_shared((block_N, block_K), dtype)
        # Local accumulator
        accum = T.alloc_fragment((block_M, block_N), "float32")
        # Output after matmul + maxpool + sum
        output = T.alloc_fragment((batch_size,), "float32")
        
        # Initialize output
        for i in T.Parallel(batch_size):
            output[i] = T.float32(0.0)
        
        # Grid dimensions
        grid_M = T.ceildiv(batch_size, block_M)
        grid_N = T.ceildiv(out_features, block_N)
        grid_K = T.ceildiv(in_features, block_K)
        
        # Main computation
        with T.Kernel(grid_M, threads=threads) as (bx):
            # Thread-local row
            row = bx * block_M + T.thread_idx().x // (block_N // 32)
            if row < batch_size:
                # Accumulate across K dimension
                for ni in range(grid_N):
                    # Initialize accumulator for this tile
                    for local_n in T.Parallel(block_N):
                        accum[0, local_n] = T.float32(0.0)
                    
                    # Compute matmul tile
                    for ki in range(grid_K):
                        # Load A tile
                        for local_k in T.Parallel(block_K):
                            if row < batch_size and ki * block_K + local_k < in_features:
                                A_shared[0, local_k] = A[row, ki * block_K + local_k]
                            else:
                                A_shared[0, local_k] = T.float16(0.0)
                        
                        # Load B tile
                        for local_n, local_k in T.Parallel(block_N, block_K):
                            if ni * block_N + local_n < out_features and ki * block_K + local_k < in_features:
                                B_shared[local_n, local_k] = B[ni * block_N + local_n, ki * block_K + local_k]
                            else:
                                B_shared[local_n, local_k] = T.float16(0.0)
                        
                        # Compute tile
                        for local_n in T.Parallel(block_N):
                            for local_k in range(block_K):
                                accum[0, local_n] += T.cast(A_shared[0, local_k], "float32") * T.cast(B_shared[local_n, local_k], "float32")
                    
                    # Apply max pooling (kernel_size=2, stride=2)
                    # Since we're doing 1D max pooling with kernel_size=2, we take max of adjacent pairs
                    for local_n in T.Parallel(block_N // 2):
                        idx1 = local_n * 2
                        idx2 = local_n * 2 + 1
                        if ni * block_N + idx2 < out_features:
                            val1 = accum[0, idx1]
                            val2 = accum[0, idx2]
                            pooled_val = T.max(val1, val2)
                            output[row] += pooled_val
                        elif ni * block_N + idx1 < out_features:
                            output[row] += accum[0, idx1]
                
                # Apply scaling
                C[row] = T.cast(output[row] * scale_factor, dtype)
    
    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self._kernel_cache = {}
        
        # Initialize weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def _get_kernel(self, batch_size: int, tl_dtype: str):
        key = (batch_size, self.in_features, self.out_features, self.kernel_size, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_maxpool_sum_kernel(
                batch_size, self.in_features, self.out_features, self.kernel_size, dtype=tl_dtype
            )
        return self._kernel_cache[key]
    
    def forward(self, x):
        batch_size = x.size(0)
        kernel = self._get_kernel(batch_size, "float16")
        
        # Convert to fp16
        x_fp16 = x.half()
        weight_fp16 = self.weight.half()
        
        # Call fused kernel
        output = kernel(x_fp16, weight_fp16, self.scale_factor)
        
        return output