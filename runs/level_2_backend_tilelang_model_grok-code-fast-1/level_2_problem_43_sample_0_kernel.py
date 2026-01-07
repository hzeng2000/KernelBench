import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_maxpool3d_kernel(B: int, C: int, D: int, H: int, W: int, block_B: int = 1, block_C: int = 1, block_D: int = 2, block_H: int = 4, block_W: int = 4, threads: int = 128, dtype: str = "float16"):
    D_out = D // 2
    H_out = H // 2
    W_out = W // 2
    
    @T.prim_func
    def maxpool3d_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        B_out: T.Tensor((B, C, D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(C, block_C), T.ceildiv(D_out, block_D), T.ceildiv(H_out, block_H), T.ceildiv(W_out, block_W), threads=threads) as (bb, bc, bd, bh, bw):
            start_b = bb * block_B
            start_c = bc * block_C
            start_d = bd * block_D
            start_h = bh * block_H
            start_w = bw * block_W

            for local_b, local_c, local_d, local_h, local_w in T.Parallel(block_B, block_C, block_D, block_H, block_W):
                b = start_b + local_b
                c = start_c + local_c
                od = start_d + local_d
                oh = start_h + local_h
                ow = start_w + local_w

                if b < B and c < C and od < D_out and oh < H_out and ow < W_out:
                    max_val = T.min_value(dtype)
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                id = od * 2 + i
                                ih = oh * 2 + j
                                iw = ow * 2 + k
                                if id < D and ih < H and iw < W:
                                    max_val = T.max(max_val, A[b, c, id, ih, iw])
                    B_out[b, c, od, oh, ow] = max_val

    return tilelang.compile(maxpool3d_kernel, out_idx=[1], target="cuda")


def build_logsumexp_relu_kernel(B: int, C: int, D: int, H: int, W: int, block_B: int = 1, block_D: int = 2, block_H: int = 4, block_W: int = 4, threads: int = 128, dtype: str = "float16"):
    
    @T.prim_func
    def logsumexp_relu_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        B_out: T.Tensor((B, 1, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(D, block_D), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bb, bd, bh, bw):
            start_b = bb * block_B
            start_d = bd * block_D
            start_h = bh * block_H
            start_w = bw * block_W

            for local_b, local_d, local_h, local_w in T.Parallel(block_B, block_D, block_H, block_W):
                b = start_b + local_b
                d = start_d + local_d
                h = start_h + local_h
                w = start_w + local_w

                if b < B and d < D and h < H and w < W:
                    max_val = T.min_value(dtype)
                    for c in range(C):
                        val = A[b, c, d, h, w]
                        max_val = T.max(max_val, val)
                    sum_exp = 0.0
                    for c in range(C):
                        val = A[b, c, d, h, w]
                        sum_exp += T.exp(val - max_val)
                    logsum = max_val + T.log(sum_exp)
                    B_out[b, 0, d, h, w] = T.max(logsum, 0.0)

    return tilelang.compile(logsumexp_relu_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, custom TileLang max pooling, and custom TileLang log sum exp + ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._kernel_cache = {}

    def _get_maxpool_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = ("maxpool", B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_maxpool3d_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_logsumexp_relu_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = ("logsumexp_relu", B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_logsumexp_relu_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, 1, depth', height', width')
        """
        x = self.conv(x)
        x = x.half()  # Cast to FP16
        B, C, D, H, W = x.shape
        kernel_maxpool = self._get_maxpool_kernel(B, C, D, H, W, "float16")
        x = kernel_maxpool(x)
        B, C, D, H, W = x.shape  # Updated shapes after maxpool
        kernel_logsumexp_relu = self._get_logsumexp_relu_kernel(B, C, D, H, W, "float16")
        x = kernel_logsumexp_relu(x)
        return x