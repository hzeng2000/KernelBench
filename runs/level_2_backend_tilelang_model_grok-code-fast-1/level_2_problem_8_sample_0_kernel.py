import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv3d_scale_kernel(N: int, C: int, D: int, H: int, W: int, K: int, KD: int, KH: int, KW: int, divisor: float, block_N: int = 1, block_K: int = 16, block_D: int = 1, block_H: int = 16, block_W: int = 16, threads: int = 256, dtype: str = "float16"):
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1
    
    @T.prim_func
    def conv3d_scale_kernel(
        Input: T.Tensor((N, C, D, H, W), dtype),
        Weight: T.Tensor((K, C, KD, KH, KW), dtype),
        Output: T.Tensor((N, K, OD, OH, OW), dtype),
    ):
        with T.Kernel(T.ceildiv(OW, block_W), T.ceildiv(OH, block_H), T.ceildiv(OD, block_D), T.ceildiv(K, block_K), T.ceildiv(N, block_N), threads=threads) as (bx, by, bz, bk, bn):
            start_n = bn * block_N
            start_k = bk * block_K
            start_d = bz * block_D
            start_h = by * block_H
            start_w = bx * block_W

            for local_n, local_k, local_d, local_h, local_w in T.Parallel(block_N, block_K, block_D, block_H, block_W):
                n = start_n + local_n
                k = start_k + local_k
                od = start_d + local_d
                oh = start_h + local_h
                ow = start_w + local_w

                if n < N and k < K and od < OD and oh < OH and ow < OW:
                    sum_val = T.float32(0)
                    for c in range(C):
                        for kd in range(KD):
                            for kh in range(KH):
                                for kw in range(KW):
                                    id = od + kd
                                    ih = oh + kh
                                    iw = ow + kw
                                    if id < D and ih < H and iw < W:
                                        sum_val += Input[n, c, id, ih, iw] * Weight[k, c, kd, kh, kw]
                    Output[n, k, od, oh, ow] = sum_val / divisor

    return tilelang.compile(conv3d_scale_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution fused with division, then applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, dtype=torch.float16)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float16))
        self.sum_dim = sum_dim
        self._kernel_cache = {}

    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, K: int, KD: int, KH: int, KW: int, divisor: float, tl_dtype: str):
        key = (N, C, D, H, W, K, KD, KH, KW, divisor, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv3d_scale_kernel(N, C, D, H, W, K, KD, KH, KW, divisor, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half()
        N, C, D, H, W = x.shape
        K = self.conv.out_channels
        KD, KH, KW = self.conv.kernel_size
        kernel = self._get_kernel(N, C, D, H, W, K, KD, KH, KW, self.divisor, "float16")
        x = kernel(x, self.conv.weight)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x