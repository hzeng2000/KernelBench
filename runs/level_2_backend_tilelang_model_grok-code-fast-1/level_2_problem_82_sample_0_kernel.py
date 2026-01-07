import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_fused_elementwise_kernel(N: int, K: int, H: int, W: int, dtype: str = "float16"):
    @T.prim_func
    def fused_elementwise_kernel(
        X: T.Tensor((N, K, H, W), dtype),
        Bias: T.Tensor((K, 1, 1), dtype),
        scale: T.float32,
        Out: T.Tensor((N, K, H, W), dtype),
    ):
        for n, k, h, w in T.Grid(N, K, H, W):
            Out[n, k, h, w] = T.tanh(X[n, k, h, w] * scale + Bias[k, 0, 0])

    return tilelang.compile(fused_elementwise_kernel, out_idx=[3], target="cuda")


def build_maxpool_kernel(N: int, K: int, H: int, W: int, OH: int, OW: int, pool: int, dtype: str = "float16"):
    @T.prim_func
    def maxpool_kernel(
        Input: T.Tensor((N, K, H, W), dtype),
        Output: T.Tensor((N, K, OH, OW), dtype),
    ):
        for n, k, oh, ow in T.Grid(N, K, OH, OW):
            max_val = T.float32(-3.4028235e+38)  # fp16 min
            for kh in T.serial(pool):
                for kw in T.serial(pool):
                    ih = oh * pool + kh
                    iw = ow * pool + kw
                    if ih < H and iw < W:
                        max_val = T.max(max_val, Input[n, k, ih, iw])
            Output[n, k, oh, ow] = max_val

    return tilelang.compile(maxpool_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).half()
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape).half())
        self.pool_kernel_size = pool_kernel_size
        self._elem_cache = {}
        self._pool_cache = {}

    def _get_elem_kernel(self, N: int, K: int, H: int, W: int):
        key = (N, K, H, W)
        if key not in self._elem_cache:
            self._elem_cache[key] = build_fused_elementwise_kernel(N, K, H, W)
        return self._elem_cache[key]

    def _get_pool_kernel(self, N: int, K: int, H: int, W: int, OH: int, OW: int, pool: int):
        key = (N, K, H, W, OH, OW, pool)
        if key not in self._pool_cache:
            self._pool_cache[key] = build_maxpool_kernel(N, K, H, W, OH, OW, pool)
        return self._pool_cache[key]

    def forward(self, x):
        x = x.half().contiguous()
        # Convolution
        x = self.conv(x)
        N, K, H, W = x.shape
        # Fused elementwise: tanh, scaling, bias addition
        kernel_elem = self._get_elem_kernel(N, K, H, W)
        x = kernel_elem(x, self.bias, self.scaling_factor)
        # Max-pooling
        OH = H // self.pool_kernel_size
        OW = W // self.pool_kernel_size
        kernel_pool = self._get_pool_kernel(N, K, H, W, OH, OW, self.pool_kernel_size)
        x = kernel_pool(x)
        return x