import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_multiply_global_avg_pool_kernel(B: int, C: int, H: int, W: int, multiplier: float, block_B: int = 1, block_C: int = 128, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def fused_multiply_global_avg_pool_kernel(
        A: T.Tensor((B, C, H, W), dtype),
        multiplier: T.Tensor((), dtype),
        C_out: T.Tensor((B, C, 1, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(C, block_C), T.ceildiv(B, block_B), threads=threads) as (bx, by):
            b_start = bx * block_B
            c_start = by * block_C

            for local_b in T.Parallel(block_B):
                for local_c in T.Parallel(block_C):
                    b = b_start + local_b
                    c = c_start + local_c

                    if b < B and c < C:
                        h_ax = T.reduce_axis(0, H)
                        w_ax = T.reduce_axis(0, W)
                        sum_val = T.reduce_sum(A[b, c, h_ax, w_ax] * multiplier, [h_ax, w_ax], init=0.0)
                        C_out[b, c, 0, 0] = sum_val / (H * W)

    return tilelang.compile(fused_multiply_global_avg_pool_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then fuses multiplication by scalar and global average pooling into a single TileLang kernel.
    The second global average pooling is redundant (as it operates on a 1x1 spatial tensor), so it is omitted.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self._kernel_cache = {}

    def _get_kernel(self, B: int, C: int, H: int, W: int, multiplier: float, tl_dtype: str):
        key = (B, C, H, W, multiplier, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_multiply_global_avg_pool_kernel(B, C, H, W, multiplier, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.to(torch.float16)  # Ensure FP16
        B, C, H, W = x.shape
        kernel = self._get_kernel(B, C, H, W, self.multiplier, "float16")
        multiplier_tensor = torch.tensor(self.multiplier, dtype=torch.float16, device=x.device)
        x = kernel(x, multiplier_tensor)
        # Second global average pooling is redundant and omitted
        return x