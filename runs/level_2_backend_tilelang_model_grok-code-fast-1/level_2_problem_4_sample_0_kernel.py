import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_mish_mish_kernel(B: int, C: int, H: int, W: int, dtype: str = "float16"):
    total_elements = B * C * H * W
    block_size = 1024

    @T.prim_func
    def mish_mish_kernel(
        A: T.Tensor((B, C, H, W), dtype),
        C_out: T.Tensor((B, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(total_elements, block_size), threads=block_size) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < total_elements:
                    b = idx // (C * H * W)
                    c = (idx % (C * H * W)) // (H * W)
                    h = (idx % (H * W)) // W
                    w = idx % W
                    x = A[b, c, h, w]
                    # First mish
                    sp = T.log(1 + T.exp(x))
                    t = T.tanh(sp)
                    x = x * t
                    # Second mish
                    sp = T.log(1 + T.exp(x))
                    t = T.tanh(sp)
                    x = x * t
                    C_out[b, c, h, w] = x

    return tilelang.compile(mish_mish_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution in FP16, and applies fused Mish twice using a custom TileLang kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, batch_size, height, width):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        oh = height - kernel_size + 1
        ow = oh
        self.kernel = build_mish_mish_kernel(batch_size, out_channels, oh, ow, "float16")

    def forward(self, x):
        x = x.half()
        x = self.conv(x)
        x = x.contiguous()
        C = torch.empty_like(x)
        self.kernel(x, C)
        return C