import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_linear_kernel(batch: int, in_feat: int, out_feat: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_linear_kernel(
        A: T.Tensor((batch, in_feat), dtype),
        W: T.Tensor((out_feat, in_feat), dtype),
        B: T.Tensor((out_feat,), dtype),
        sub: T.float32,
        mul: T.float32,
        C: T.Tensor((batch, out_feat), dtype),
    ):
        with T.Kernel(T.ceildiv(batch, block_M), T.ceildiv(out_feat, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(in_feat, block_K), num_stages=3):
                T.copy(A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(W[by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K], W_shared)
                
                for i, j, kk in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, kk] * W_shared[j, kk]
            
            for i, j in T.Parallel(block_M, block_N):
                c_val = C_local[i, j]
                c_val += B[by * block_N + j]
                c_val -= sub
                c_val *= mul
                C[bx * block_M + i, by * block_N + j] = T.max(c_val, T.float16(0))
    
    return tilelang.compile(fused_linear_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16))
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self._kernel_cache = {}

    def _get_kernel(self, batch: int, in_feat: int, out_feat: int, tl_dtype: str):
        key = (batch, in_feat, out_feat, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_linear_kernel(batch, in_feat, out_feat, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.contiguous().half()
        batch, in_feat = x.shape
        out_feat = self.weight.shape[0]
        kernel = self._get_kernel(batch, in_feat, out_feat, "float16")
        C = kernel(x, self.weight, self.bias, self.subtract_value, self.multiply_value)
        return C