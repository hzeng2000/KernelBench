import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(M: int, N: int, block_M: int = 128, block_N: int = 64, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for local_y, local_x in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                if y < M and x < N:
                    tanh_x = T.tanh(B[y, x])
                    relu6 = T.relu(T.min(T.max(tanh_x + T.cast(3.0, dtype), T.cast(0.0, dtype)), T.cast(6.0, dtype)))
                    hardswish = tanh_x * relu6 / T.cast(6.0, dtype)
                    C[y, x] = A[y, x] + hardswish

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


def build_logsumexp_kernel(M: int, N: int, block_M: int = 128, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def logsumexp_kernel(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as bx:
            start_y = bx * block_M

            for local_y in T.Parallel(block_M):
                y = start_y + local_y

                if y < M:
                    max_val = T.cast(-float('inf'), dtype)
                    for x in range(N):
                        max_val = T.max(max_val, A[y, x])
                    
                    sum_exp = T.cast(0.0, dtype)
                    for x in range(N):
                        sum_exp += T.exp(A[y, x] - max_val)
                    
                    C[y, 0] = T.log(sum_exp) + max_val

    return tilelang.compile(logsumexp_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, applies Group Normalization, fused Tanh+HardSwish+Residual Addition, and LogSumExp with TileLang kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self._fused_kernel_cache = {}
        self._logsumexp_kernel_cache = {}

    def _get_fused_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._fused_kernel_cache:
            self._fused_kernel_cache[key] = build_fused_kernel(M, N, dtype=tl_dtype)
        return self._fused_kernel_cache[key]

    def _get_logsumexp_kernel(self, M: int, N: int, tl_dtype: str):
        key = (M, N, tl_dtype)
        if key not in self._logsumexp_kernel_cache:
            self._logsumexp_kernel_cache[key] = build_logsumexp_kernel(M, N, dtype=tl_dtype)
        return self._logsumexp_kernel_cache[key]

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x).half()  # Convert to FP16
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        
        # Prepare for fused kernel: flatten to (M, N)
        batch_size, channels, height, width = x_conv.shape
        M = batch_size * height * width
        N = channels
        x_conv_flat = x_conv.view(M, N)
        x_norm_flat = x_norm.view(M, N)
        
        # Fused Tanh + HardSwish + Residual Addition
        fused_kernel = self._get_fused_kernel(M, N, "float16")
        x_res_flat = fused_kernel(x_conv_flat, x_norm_flat)
        x_res = x_res_flat.view(batch_size, channels, height, width)
        
        # LogSumExp
        x_res_flat = x_res.view(M, N)
        logsumexp_kernel = self._get_logsumexp_kernel(M, N, "float16")
        x_out_flat = logsumexp_kernel(x_res_flat)
        x_out = x_out_flat.view(batch_size, 1, height, width)
        
        return x_out