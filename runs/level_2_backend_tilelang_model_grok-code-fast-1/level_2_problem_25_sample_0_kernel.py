import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_min_tanh_kernel(M: int, C: int, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def min_tanh_kernel(
        A: T.Tensor((M, C), dtype),
        B: T.Tensor((M, 1), dtype),
    ):
        with T.Kernel(M, threads=threads) as i:
            min_val = T.reduce(lambda j: A[i, j], T.reduce_axis(0, C), T.min, init=T.float32(float('inf')))
            B[i, 0] = T.tanh(T.tanh(min_val))

    return tilelang.compile(min_tanh_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._kernel_cache = {}

    def _get_kernel(self, M: int, C: int, tl_dtype: str):
        key = (M, C, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_min_tanh_kernel(M, C, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        kernel = self._get_kernel(B * H * W, C, "float16")
        out = kernel(x)
        out = out.view(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
        return out