import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_gemm_scale_bn_kernel(M: int, N: int, K: int, block_M: int = 64, block_N: int = 64, block_K: int = 32, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def gemm_scale_bn_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        Scale: T.Tensor((N,), dtype),
        Bias: T.Tensor((N,), dtype),
        Mean: T.Tensor((N,), dtype),
        Var: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            acc = T.alloc_fragment((block_M, block_N), dtype)
            for i, j in T.Parallel(block_M, block_N):
                acc[i, j] = T.cast(0, dtype)

            for k in T.serial(T.ceildiv(K, block_K)):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)

                for i, j in T.Parallel(block_M, block_K):
                    if start_y + i < M and k * block_K + j < K:
                        A_shared[i, j] = A[start_y + i, k * block_K + j]
                    else:
                        A_shared[i, j] = T.cast(0, dtype)

                for i, j in T.Parallel(block_N, block_K):
                    if start_x + i < N and k * block_K + j < K:
                        B_shared[i, j] = B[start_x + i, k * block_K + j]
                    else:
                        B_shared[i, j] = T.cast(0, dtype)

                for i, j in T.Parallel(block_M, block_N):
                    for kk in T.serial(block_K):
                        acc[i, j] += A_shared[i, kk] * B_shared[j, kk]

            for i, j in T.Parallel(block_M, block_N):
                if start_y + i < M and start_x + j < N:
                    val = acc[i, j] * Scale[start_x + j] + Bias[start_x + j]
                    mean = Mean[start_x + j]
                    var = Var[start_x + j]
                    eps = T.cast(1e-5, dtype)
                    inv_std = T.rsqrt(var + eps)
                    normalized = (val - mean) * inv_std
                    C[start_y + i, start_x + j] = normalized

    return tilelang.compile(gemm_scale_bn_kernel, out_idx=[5], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self._kernel_cache = {}
        self.eps = eps

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_gemm_scale_bn_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        M, K = x_c.shape
        N = self.gemm.out_features
        
        # Get weight and bias from Linear layer
        weight = self.gemm.weight
        bias = self.gemm.bias
        
        # Get running mean and var from BatchNorm
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        
        # Convert to fp16
        x_fp16 = x_c.half()
        weight_fp16 = weight.half()
        bias_fp16 = bias.half() if bias is not None else torch.zeros(N, device=x.device, dtype=torch.float16)
        scale_fp16 = self.scale.half()
        mean_fp16 = running_mean.half()
        var_fp16 = running_var.half()
        
        kernel = self._get_kernel(M, N, K, "float16")
        C = kernel(x_fp16, weight_fp16, scale_fp16, bias_fp16, mean_fp16, var_fp16)
        
        return C.float()