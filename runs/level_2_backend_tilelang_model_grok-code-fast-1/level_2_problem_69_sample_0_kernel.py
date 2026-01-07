import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_conv_hardswish_relu_kernel(batch: int, in_c: int, out_c: int, h: int, w: int, k: int, dtype: str = "float16"):
    oh = h - k + 1
    ow = w - k + 1
    
    @T.prim_func
    def fused_conv_hardswish_relu_kernel(
        A: T.Tensor((batch, in_c, h, w), dtype),
        W: T.Tensor((out_c, in_c, k, k), dtype),
        C: T.Tensor((batch, out_c, oh, ow), dtype),
    ):
        with T.Kernel(batch, out_c, oh, ow, threads=256) as (b, oc, i, j):
            acc = T.alloc_fragment((1,), dtype, "local")
            acc[0] = T.cast(0.0, dtype)
            for ic in range(in_c):
                for kh in range(k):
                    for kw in range(k):
                        acc[0] += A[b, ic, i + kh, j + kw] * W[oc, ic, kh, kw]
            x = acc[0]
            clamped = T.clamp(x + T.cast(3.0, dtype), T.cast(0.0, dtype), T.cast(6.0, dtype))
            hardswish = x * clamped / T.cast(6.0, dtype)
            C[b, oc, i, j] = T.max(hardswish, T.cast(0.0, dtype))
    
    return tilelang.compile(fused_conv_hardswish_relu_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._kernel_cache = {}
    
    def _get_kernel(self, batch: int, in_c: int, out_c: int, h: int, w: int, k: int, tl_dtype: str):
        key = (batch, in_c, out_c, h, w, k, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv_hardswish_relu_kernel(batch, in_c, out_c, h, w, k, dtype=tl_dtype)
        return self._kernel_cache[key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        w = self.conv.weight.contiguous()
        
        batch, _, h, w = x_c.shape
        _, out_c, k, _ = w.shape
        in_c = w.shape[1]
        
        kernel = self._get_kernel(batch, in_c, out_c, h, w, k, "float16")
        C = kernel(x_c, w)
        
        return C