import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_matmul_swish_scale_kernel(M: int, K: int, N: int, block_M: int = 128, block_K: int = 32, block_N: int = 128, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_matmul_swish_scale_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        B_bias: T.Tensor((N,), dtype),
        scale: T.Tensor((), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            T.use_swizzle(panel_size=10, enable=1)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                T.sync()
                
                for i, j, l in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, l] * B_shared[j, l]
                
                T.sync()
            
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] += B_bias[bx * block_N + j]
                y = C_local[i, j]
                sig = 1.0 / (1.0 + T.exp(-y))
                C_local[i, j] = y * sig * scale[()]
            
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return tilelang.compile(fused_matmul_swish_scale_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, M: int, K: int, N: int, tl_dtype: str):
        key = (M, K, N, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_matmul_swish_scale_kernel(M, K, N, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x_c = x.contiguous()
        weight_c = self.matmul.weight.contiguous()
        bias_c = self.matmul.bias.contiguous()
        scale_tensor = torch.tensor(self.scaling_factor, dtype=torch.float16, device=x.device)
        
        M, K = x_c.shape
        N = weight_c.shape[0]
        
        kernel = self._get_kernel(M, K, N, "float16")
        C = kernel(x_c, weight_c, bias_c, scale_tensor)
        
        return C