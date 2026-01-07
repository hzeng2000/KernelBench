import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_conv_gelu_kernel(batch: int, in_c: int, out_c: int, H: int, W: int, k: int, dtype: str = "float16"):
    H_out = H - k + 1
    W_out = W - k + 1
    block_b = 1
    block_oc = 16
    block_oh = 8
    block_ow = 8
    threads = 128

    @T.prim_func
    def conv_gelu_kernel(
        X: T.Tensor((batch, in_c, H, W), dtype),
        W: T.Tensor((out_c, in_c, k, k), dtype),
        Y: T.Tensor((batch, out_c, H_out, W_out), dtype),
    ):
        with T.Kernel(T.ceildiv(batch, block_b), T.ceildiv(out_c, block_oc), T.ceildiv(H_out, block_oh), T.ceildiv(W_out, block_ow), threads=threads) as (bb, boc, boh, bow):
            start_b = bb * block_b
            start_oc = boc * block_oc
            start_oh = boh * block_oh
            start_ow = bow * block_ow
            for local_b in T.Parallel(block_b):
                for local_oc in T.Parallel(block_oc):
                    for local_oh in T.Parallel(block_oh):
                        for local_ow in T.Parallel(block_ow):
                            b = start_b + local_b
                            oc = start_oc + local_oc
                            oh = start_oh + local_oh
                            ow = start_ow + local_ow
                            if b < batch and oc < out_c and oh < H_out and ow < W_out:
                                sum_val = T.cast(0.0, dtype)
                                for ic in range(in_c):
                                    for kh in range(k):
                                        for kw in range(k):
                                            sum_val += X[b, ic, oh + kh, ow + kw] * W[oc, ic, kh, kw]
                                x = sum_val
                                y = x * 0.5 * (1.0 + T.erf(x * T.rsqrt(2.0)))
                                Y[b, oc, oh, ow] = y

    return tilelang.compile(conv_gelu_kernel, out_idx=[2], target="cuda")


def build_avg_pool_kernel(batch: int, out_c: int, H_out: int, W_out: int, dtype: str = "float16"):
    block_b = 1
    block_oc = 32
    threads = 32

    @T.prim_func
    def avg_pool_kernel(
        Y: T.Tensor((batch, out_c, H_out, W_out), dtype),
        Z: T.Tensor((batch, out_c), dtype),
    ):
        with T.Kernel(T.ceildiv(batch, block_b), T.ceildiv(out_c, block_oc), threads=threads) as (bb, boc):
            start_b = bb * block_b
            start_oc = boc * block_oc
            for local_b in T.Parallel(block_b):
                for local_oc in T.Parallel(block_oc):
                    b = start_b + local_b
                    oc = start_oc + local_oc
                    if b < batch and oc < out_c:
                        sum_val = T.cast(0.0, dtype)
                        for oh in range(H_out):
                            for ow in range(W_out):
                                sum_val += Y[b, oc, oh, ow]
                        Z[b, oc] = sum_val / T.cast(H_out * W_out, dtype)

    return tilelang.compile(avg_pool_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._kernel_cache1 = {}
        self._kernel_cache2 = {}

    def _get_kernel1(self, batch: int, in_c: int, out_c: int, H: int, W: int, k: int, tl_dtype: str):
        key = (batch, in_c, out_c, H, W, k, tl_dtype)
        if key not in self._kernel_cache1:
            self._kernel_cache1[key] = build_conv_gelu_kernel(batch, in_c, out_c, H, W, k, tl_dtype)
        return self._kernel_cache1[key]

    def _get_kernel2(self, batch: int, out_c: int, H_out: int, W_out: int, tl_dtype: str):
        key = (batch, out_c, H_out, W_out, tl_dtype)
        if key not in self._kernel_cache2:
            self._kernel_cache2[key] = build_avg_pool_kernel(batch, out_c, H_out, W_out, tl_dtype)
        return self._kernel_cache2[key]

    def forward(self, x):
        x = x.half()
        weight = self.conv.weight.half()
        batch, in_c, H, W = x.shape
        out_c = weight.shape[0]
        k = weight.shape[2]
        H_out = H - k + 1
        W_out = W - k + 1
        kernel1 = self._get_kernel1(batch, in_c, out_c, H, W, k, "float16")
        Y = kernel1(x, weight)
        kernel2 = self._get_kernel2(batch, out_c, H_out, W_out, "float16")
        Z = kernel2(Y)
        return Z.float()