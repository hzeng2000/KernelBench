import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_post_pool_kernel(B: int, C: int, D: int, H: int, W: int, block_B: int = 1, block_D: int = 2, block_H: int = 4, block_W: int = 4, threads: int = 32, dtype: str = "float16"):
    
    @T.prim_func
    def fused_post_pool_kernel(
        pooled: T.Tensor((B, C, D, H, W), dtype),
        subtract: T.Tensor((C,), dtype),
        output: T.Tensor((B, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(W, block_W), T.ceildiv(H, block_H), T.ceildiv(D, block_D), T.ceildiv(B, block_B), threads=threads) as (bx, by, bz, bb):
            start_w = bx * block_W
            start_h = by * block_H
            start_d = bz * block_D
            start_b = bb * block_B

            for local_b, local_d, local_h, local_w in T.Parallel(block_B, block_D, block_H, block_W):
                b = start_b + local_b
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w

                if b < B and d < D and h < H and w < W:
                    # Compute max over channels
                    max_val = T.float16(-1e10)
                    for c in range(C):
                        val = pooled[b, c, d, h, w]
                        max_val = T.max(max_val, val)
                    
                    # Compute sum of exp
                    sum_exp = T.float16(0)
                    for c in range(C):
                        val = pooled[b, c, d, h, w]
                        sum_exp += T.exp(val - max_val)
                    
                    # Compute max of swish over channels
                    final_max = T.float16(-1e10)
                    for c in range(C):
                        val = pooled[b, c, d, h, w]
                        softmax_val = T.exp(val - max_val) / sum_exp
                        sub_val = softmax_val - subtract[c]
                        swish_val = T.sigmoid(sub_val) * sub_val
                        final_max = T.max(final_max, swish_val)
                    
                    output[b, d, h, w] = final_max

    return tilelang.compile(fused_post_pool_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model with fused TileLang kernel for post-pool operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels).half())  # FP16
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = (B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_post_pool_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half()  # Convert to FP16
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = x.contiguous()
        
        B, C, D, H, W = x.shape
        kernel = self._get_kernel(B, C, D, H, W, "float16")
        output = kernel(x, self.subtract)
        
        return output