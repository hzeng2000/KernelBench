import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_gemm_scale_kernel(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128, block_K: int = 32, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_gemm_scale_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        Bias: T.Tensor((N,), dtype),
        Scale: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            T.clear(C_local)
            
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(X[by * block_M : (by + 1) * block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(W[bx * block_N : (bx + 1) * block_N, k * block_K : (k + 1) * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            T.parallel(block_M, block_N, lambda i, j: (
                C_local[i, j] := C_local[i, j] + Bias[bx * block_N + j],
                C_local[i, j] := C_local[i, j] * Scale[bx * block_N + j]
            ))
            
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])
    
    return tilelang.compile(fused_gemm_scale_kernel, out_idx=[4], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, bias addition, and scaling into a single TileLang kernel,
    followed by standard BatchNorm1d.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_shape = scale_shape
        # Initialize parameters as in original
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_gemm_scale_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x_c = x.contiguous()
        original_shape = x_c.shape
        x_c = x_c.view(-1, self.in_features)
        
        M, K = x_c.shape
        N = self.out_features
        kernel = self._get_kernel(M, N, K, "float16")
        # Convert to half for FP16
        x_half = x_c.half()
        w_half = self.gemm_weight.half()
        b_half = self.gemm_bias.half()
        s_half = self.scale.half()
        C = kernel(x_half, w_half, b_half, s_half)
        
        # Apply BatchNorm
        C = self.bn(C.float())  # BatchNorm expects float, convert back
        
        return C.view(original_shape).half()  # Return as half if needed, but original is float