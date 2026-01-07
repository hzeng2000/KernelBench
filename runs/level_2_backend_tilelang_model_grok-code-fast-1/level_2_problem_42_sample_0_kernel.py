import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_v_kernel(B: int, C: int, H: int, W: int, block_B: int = 1, block_C: int = 32, threads: int = 128):
    @T.prim_func
    def v_kernel(
        x: T.Tensor((B, C, H, W), "float16"),
        bias: T.Tensor((C, 1, 1), "float16"),
        v: T.Tensor((B, C), "float16"),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(C, block_C), threads=threads) as (bx, by):
            for local_b, local_c in T.Parallel(block_B, block_C):
                b = bx * block_B + local_b
                c = by * block_C + local_c
                if b < B and c < C:
                    sum_val = T.reduce(T.ReduceOp.SUM, [T.reduce_axis(0, H), T.reduce_axis(0, W)], x[b, c, T.reduce_axis(0, H), T.reduce_axis(0, W)], init=0.0)
                    v[b, c] = sum_val / (H * W) + bias[c, 0, 0]
    return tilelang.compile(v_kernel, out_idx=[2], target="cuda")


def build_logsumexp_kernel(B: int, C: int, block_B: int = 1, threads: int = 128):
    @T.prim_func
    def logsumexp_kernel(
        v: T.Tensor((B, C), "float16"),
        y: T.Tensor((B,), "float16"),
    ):
        with T.Kernel(T.ceildiv(B, block_B), 1, threads=threads) as (bx, _):
            for local_b in T.Parallel(block_B):
                b = bx * block_B + local_b
                if b < B:
                    max_v = T.reduce(T.ReduceOp.MAX, [T.reduce_axis(0, C)], v[b, T.reduce_axis(0, C)], init=-float('inf'))
                    sum_exp = T.reduce(T.ReduceOp.SUM, [T.reduce_axis(0, C)], T.exp(v[b, T.reduce_axis(0, C)] - max_v), init=0.0)
                    y[b] = (max_v + T.log(sum_exp)) * 10.0
    return tilelang.compile(logsumexp_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then uses custom TileLang kernels for global average pooling, bias addition, log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._kernel_cache = {}

    def _get_v_kernel(self, B: int, C: int, H: int, W: int):
        key = ("v", B, C, H, W)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_v_kernel(B, C, H, W)
        return self._kernel_cache[key]

    def _get_logsumexp_kernel(self, B: int, C: int):
        key = ("logsumexp", B, C)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_logsumexp_kernel(B, C)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, H, W = x.shape
        v_kernel = self._get_v_kernel(B, C, H, W)
        v = torch.empty(B, C, dtype=torch.float16, device=x.device)
        v_kernel(x.to(torch.float16), self.bias.to(torch.float16), v)
        logsumexp_kernel = self._get_logsumexp_kernel(B, C)
        y = torch.empty(B, dtype=torch.float16, device=x.device)
        logsumexp_kernel(v, y)
        return y.to(torch.float32)