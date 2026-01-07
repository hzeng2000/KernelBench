import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_maxpool1_kernel(B: int, C: int, D: int, H: int, W: int, block_size: int = 8, threads: int = 128, dtype: str = "float16"):
    OD = D // 2
    OH = H // 2
    OW = W // 2

    @T.prim_func
    def maxpool1_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        B_out: T.Tensor((B, C, OD, OH, OW), dtype),
    ):
        with T.Kernel(T.ceildiv(OW, block_size), T.ceildiv(OH, block_size), T.ceildiv(OD, block_size), T.ceildiv(B, 1), threads=threads) as (bx, by, bz, bb):
            start_ow = bx * block_size
            start_oh = by * block_size
            start_od = bz * block_size
            start_b = bb * 1

            for local_od, local_oh, local_ow in T.Parallel(block_size, block_size, block_size):
                od = start_od + local_od
                oh = start_oh + local_oh
                ow = start_ow + local_ow
                b = start_b

                if od < OD and oh < OH and ow < OW and b < B:
                    for c in range(C):
                        max_val = T.cast(-float('inf'), dtype)
                        for dd in range(2):
                            for hh in range(2):
                                for ww in range(2):
                                    id = od * 2 + dd
                                    ih = oh * 2 + hh
                                    iw = ow * 2 + ww
                                    if id < D and ih < H and iw < W:
                                        max_val = T.max(max_val, A[b, c, id, ih, iw])
                        B_out[b, c, od, oh, ow] = max_val

    return tilelang.compile(maxpool1_kernel, out_idx=[1], target="cuda")


def build_fused_maxpool2_sum_kernel(B: int, C: int, D: int, H: int, W: int, block_size: int = 8, threads: int = 128, dtype: str = "float16"):
    OD = D // 3
    OH = H // 3
    OW = W // 3

    @T.prim_func
    def fused_maxpool2_sum_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        B_out: T.Tensor((B, 1, OD, OH, OW), dtype),
    ):
        with T.Kernel(T.ceildiv(OW, block_size), T.ceildiv(OH, block_size), T.ceildiv(OD, block_size), T.ceildiv(B, 1), threads=threads) as (bx, by, bz, bb):
            start_ow = bx * block_size
            start_oh = by * block_size
            start_od = bz * block_size
            start_b = bb * 1

            for local_od, local_oh, local_ow in T.Parallel(block_size, block_size, block_size):
                od = start_od + local_od
                oh = start_oh + local_oh
                ow = start_ow + local_ow
                b = start_b

                if od < OD and oh < OH and ow < OW and b < B:
                    sum_val = T.cast(0.0, dtype)
                    for c in range(C):
                        max_val = T.cast(-float('inf'), dtype)
                        for dd in range(3):
                            for hh in range(3):
                                for ww in range(3):
                                    id = od * 3 + dd
                                    ih = oh * 3 + hh
                                    iw = ow * 3 + ww
                                    if id < D and ih < H and iw < W:
                                        max_val = T.max(max_val, A[b, c, id, ih, iw])
                        sum_val += max_val
                    B_out[b, 0, od, oh, ow] = sum_val

    return tilelang.compile(fused_maxpool2_sum_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._kernel_cache = {}

    def _get_maxpool1_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = ("maxpool1", B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_maxpool1_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_fused_kernel(self, B: int, C: int, D: int, H: int, W: int, tl_dtype: str):
        key = ("fused", B, C, D, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_maxpool2_sum_kernel(B, C, D, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.half().contiguous()
        B, C, D, H, W = x.shape
        maxpool1_kernel = self._get_maxpool1_kernel(B, C, D, H, W, "float16")
        x = maxpool1_kernel(x)
        B, C, D, H, W = x.shape
        fused_kernel = self._get_fused_kernel(B, C, D, H, W, "float16")
        x = fused_kernel(x)
        return x.float()