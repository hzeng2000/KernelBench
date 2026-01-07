import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_avgpool_sigmoid_kernel(batch: int = 128, out_c: int = 64, H: int = 382, W: int = 382, pool_h: int = 95, pool_w: int = 95, pool_k: int = 4, dtype: str = "float16"):
    
    @T.prim_func
    def fused_avgpool_sigmoid_kernel(
        D: T.Tensor((batch, out_c, H, W), dtype),
        E: T.Tensor((batch, out_c, pool_h, pool_w), dtype),
    ):
        with T.Kernel(batch, out_c, threads=128) as (bx, by):
            for hh in T.serial(pool_h):
                for ww in T.serial(pool_w):
                    avg = T.alloc_var(dtype, shape=())
                    avg[()] = 0.0
                    for kh in T.serial(pool_k):
                        for kw in T.serial(pool_k):
                            avg[()] += D[bx, by, hh * pool_k + kh, ww * pool_k + kw]
                    avg[()] = avg[()] / (pool_k * pool_k)
                    E[bx, by, hh, ww] = 1.0 / (1.0 + T.exp(-avg[()]))

    return tilelang.compile(fused_avgpool_sigmoid_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    This model performs a convolution, fused average pooling with sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.kernel = build_fused_avgpool_sigmoid_kernel()

    def forward(self, x):
        x = x.half()
        x = self.conv(x)
        x = self.kernel(x)
        x = torch.sum(x, dim=[1, 2, 3])
        return x.float()