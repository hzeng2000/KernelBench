import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(B: int, C: int, H: int, W: int, block_H: int = 16, block_W: int = 16, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        A: T.Tensor((B, C, H, W), dtype),
        Bias: T.Tensor((C, 1, 1), dtype),
        Scaling: T.Tensor((), dtype),
        Out: T.Tensor((B, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(W, block_W), T.ceildiv(H, block_H), threads=threads) as (bx, by):
            start_w = bx * block_W
            start_h = by * block_H

            for local_h, local_w in T.Parallel(block_H, block_W):
                h = start_h + local_h
                w = start_w + local_w

                if h < H and w < W:
                    for b in range(B):
                        for c in range(C):
                            val = A[b, c, h, w] + Bias[c, 0, 0]
                            val = T.clamp(val, T.float32(0.0), T.float32(1.0))
                            val = val * Scaling[()]
                            val = T.clamp(val, T.float32(0.0), T.float32(1.0))
                            val = val / Scaling[()]
                            Out[b, c, h, w] = val

    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then fuses the bias add, clamps, scaling, and division into a single TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, H: int, W: int, tl_dtype: str):
        key = (B, C, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(B, C, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x_c = x.contiguous()
        B, C, H, W = x_c.shape
        bias_c = self.bias.contiguous()
        scaling_c = torch.tensor(self.scaling_factor, dtype=torch.float16, device=x.device)
        kernel = self._get_kernel(B, C, H, W, "float16")
        out = kernel(x_c, bias_c, scaling_c)
        return out