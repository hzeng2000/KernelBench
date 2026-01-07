import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_kernel(B: int, C: int, D: int, H: int, W: int, block_B: int = 1, block_C: int = 1, block_D: int = 4, block_H: int = 8, block_W: int = 8, threads: int = 256, dtype: str = "float16"):
    
    @T.prim_func
    def fused_kernel(
        Y: T.Tensor((B, C, D, H, W), dtype),
        Bias: T.Tensor((C,), dtype),
        Z: T.Tensor((B, C, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(C, block_C), T.ceildiv(D, block_D), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bb, bc, bd, bh, bw):
            for local_bb, local_bc, local_bd, local_bh, local_bw in T.Parallel(block_B, block_C, block_D, block_H, block_W):
                b = bb * block_B + local_bb
                c = bc * block_C + local_bc
                d = bd * block_D + local_bd
                h = bh * block_H + local_bh
                w = bw * block_W + local_bw
                
                if b < B and c < C and d < D and h < H and w < W:
                    y_val = Y[b, c, d, h, w]
                    bias_val = Bias[c]
                    Z[b, c, d, h, w] = 2 * y_val * y_val + (bias_val + 1) * y_val

    return tilelang.compile(fused_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by fused element-wise operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = (B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        bias_c = self.bias.squeeze().contiguous()
        
        B, C, D, H, W = x.shape
        kernel = self._get_kernel(B, C, D, H, W, "float16")
        z = kernel(x, bias_c)
        
        return z