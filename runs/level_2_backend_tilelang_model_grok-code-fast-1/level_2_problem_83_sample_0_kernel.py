import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_clamp_kernel(M: int, block_size: int = 1024, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def clamp_kernel(
        A: T.Tensor((M,), dtype),
        min_val: T.float16,
        max_val: T.float16,
        C: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_size), threads=threads) as bx:
            start = bx * block_size
            for i in T.Parallel(block_size):
                idx = start + i
                if idx < M:
                    C[idx] = T.min(T.max(A[idx], min_val), max_val)

    return tilelang.compile(clamp_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Group Normalization, fused min and clamp with custom TileLang kernel, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.min_value = min_value
        self.max_value = max_value
        self._kernel_cache = {}

    def _get_kernel(self, M: int, tl_dtype: str):
        key = (M, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_clamp_kernel(M, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half()
        x = self.conv(x)
        x = self.norm(x)
        
        # Flatten for element-wise operation
        original_shape = x.shape
        x_flat = x.view(-1)
        
        M = x_flat.numel()
        kernel = self._get_kernel(M, "float16")
        x_flat = kernel(x_flat, self.min_value, self.max_value)
        
        x = x_flat.view(original_shape)
        x = self.dropout(x)
        return x