import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_min_sub_kernel(
    batch: int,
    in_f: int,
    out_f: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    threads: int = 128,
    dtype: str = "float16"
):
    num_k_blocks = T.ceildiv(in_f, block_K)
    
    @T.prim_func
    def fused_linear_min_sub_kernel(
        A: T.Tensor((batch, in_f), dtype),
        B: T.Tensor((out_f, in_f), dtype),
        bias: T.Tensor((out_f,), dtype),
        constant: T.Tensor((), dtype),
        C: T.Tensor((batch, out_f), dtype),
    ):
        with T.Kernel(T.ceildiv(out_f, block_N), T.ceildiv(batch, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            start_y = by * block_M
            start_x = bx * block_N
            
            # Initialize C_local with bias
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                if y < batch and x < out_f:
                    C_local[local_y, local_x] = bias[x]
                else:
                    C_local[local_y, local_x] = T.float16(0)
            
            # Main computation loop
            for k_block in T.Pipeline(num_k_blocks):
                start_k = k_block * block_K
                
                # Load A_shared
                for local_y, local_k in T.Parallel(block_M, block_K):
                    y = start_y + local_y
                    k = start_k + local_k
                    if y < batch and k < in_f:
                        A_shared[local_y, local_k] = A[y, k]
                    else:
                        A_shared[local_y, local_k] = T.float16(0)
                
                # Load B_shared
                for local_x, local_k in T.Parallel(block_N, block_K):
                    x = start_x + local_x
                    k = start_k + local_k
                    if x < out_f and k < in_f:
                        B_shared[local_x, local_k] = B[x, k]
                    else:
                        B_shared[local_x, local_k] = T.float16(0)
                
                # Compute
                for local_y, local_x, local_k in T.Parallel(block_M, block_N, block_K):
                    C_local[local_y, local_x] += A_shared[local_y, local_k] * B_shared[local_x, local_k]
            
            # Apply min and subtract
            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x
                if y < batch and x < out_f:
                    temp = T.min(C_local[local_y, local_x], constant)
                    C[y, x] = temp - constant
    
    return tilelang.compile(fused_linear_min_sub_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self.constant = nn.Parameter(torch.tensor(constant, dtype=torch.float16))
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, in_f: int, out_f: int, tl_dtype: str):
        key = (batch, in_f, out_f, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_min_sub_kernel(batch, in_f, out_f, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        
        batch, in_f = x_c.shape
        out_f = self.weight.shape[0]
        
        kernel = self._get_kernel(batch, in_f, out_f, "float16")
        C = kernel(x_c, self.weight, self.bias, self.constant)
        
        return C