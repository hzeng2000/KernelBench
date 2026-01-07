import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_subtract_hardswish_kernel(B: int, C: int, H: int, W: int, block_BC: int = 1, block_H: int = 8, block_W: int = 8, threads: int = 64, dtype: str = "float16"):
    
    @T.prim_func
    def subtract_hardswish_kernel(
        A: T.Tensor((B, C, H, W), dtype),
        subtract_val: T.Tensor((), dtype),
        C_out: T.Tensor((B, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(W, block_W), T.ceildiv(H, block_H), T.ceildiv(B*C, block_BC), threads=threads) as (bx, by, bz):
            start_w = bx * block_W
            start_h = by * block_H
            start_bc = bz * block_BC

            for local_bc, local_h, local_w in T.Parallel(block_BC, block_H, block_W):
                bc = start_bc + local_bc
                h = start_h + local_h
                w = start_w + local_w

                if bc < B * C and h < H and w < W:
                    b = bc // C
                    c = bc % C
                    val = A[b, c, h, w] - subtract_val[()]
                    clamped = T.clamp(val + 3.0, 0.0, 6.0)
                    C_out[b, c, h, w] = val * clamped / 6.0

    return tilelang.compile(subtract_hardswish_kernel, out_idx=[2], target="cuda")


def build_maxpool_kernel(B: int, C: int, H: int, W: int, kernel_size: int = 2, stride: int = 2, block_BC: int = 1, block_H: int = 4, block_W: int = 4, threads: int = 16, dtype: str = "float16"):
    
    @T.prim_func
    def maxpool_kernel(
        A: T.Tensor((B, C, H, W), dtype),
        C_out: T.Tensor((B, C, H//stride, W//stride), dtype),
    ):
        with T.Kernel(T.ceildiv(W//stride, block_W), T.ceildiv(H//stride, block_H), T.ceildiv(B*C, block_BC), threads=threads) as (bx, by, bz):
            start_w = bx * block_W
            start_h = by * block_H
            start_bc = bz * block_BC

            for local_bc, local_h, local_w in T.Parallel(block_BC, block_H, block_W):
                bc = start_bc + local_bc
                h_out = start_h + local_h
                w_out = start_w + local_w

                if bc < B * C and h_out < H//stride and w_out < W//stride:
                    b = bc // C
                    c = bc % C
                    max_val = T.min_value(dtype)
                    for kh in T.serial(kernel_size):
                        for kw in T.serial(kernel_size):
                            h_in = stride * h_out + kh
                            w_in = stride * w_out + kw
                            if h_in < H and w_in < W:
                                max_val = T.max(max_val, A[b, c, h_in, w_in])
                    C_out[b, c, h_out, w_out] = max_val

    return tilelang.compile(maxpool_kernel, out_idx=[1], target="cuda")


def build_mish_kernel(B: int, C: int, H: int, W: int, block_BC: int = 1, block_H: int = 8, block_W: int = 8, threads: int = 64, dtype: str = "float16"):
    
    @T.prim_func
    def mish_kernel(
        A: T.Tensor((B, C, H, W), dtype),
        C_out: T.Tensor((B, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(W, block_W), T.ceildiv(H, block_H), T.ceildiv(B*C, block_BC), threads=threads) as (bx, by, bz):
            start_w = bx * block_W
            start_h = by * block_H
            start_bc = bz * block_BC

            for local_bc, local_h, local_w in T.Parallel(block_BC, block_H, block_W):
                bc = start_bc + local_bc
                h = start_h + local_h
                w = start_w + local_w

                if bc < B * C and h < H and w < W:
                    b = bc // C
                    c = bc % C
                    val = A[b, c, h, w]
                    softplus = T.log(1.0 + T.exp(val))
                    tanh_softplus = T.tanh(softplus)
                    C_out[b, c, h, w] = val * tanh_softplus

    return tilelang.compile(mish_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions using custom TileLang kernels for FP16.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, dtype=torch.float16)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        self._kernel_cache = {}

    def _get_subtract_hardswish_kernel(self, B: int, C: int, H: int, W: int, tl_dtype: str):
        key = ("subtract_hardswish", B, C, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_subtract_hardswish_kernel(B, C, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_maxpool_kernel(self, B: int, C: int, H: int, W: int, tl_dtype: str):
        key = ("maxpool", B, C, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_maxpool_kernel(B, C, H, W, kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size, dtype=tl_dtype)
        return self._kernel_cache[key]

    def _get_mish_kernel(self, B: int, C: int, H: int, W: int, tl_dtype: str):
        key = ("mish", B, C, H, W, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_mish_kernel(B, C, H, W, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x):
        x = x.half()
        x = self.conv(x)
        B, C, H, W = x.shape
        kernel = self._get_subtract_hardswish_kernel(B, C, H, W, "float16")
        subtract_val_tensor = torch.tensor(self.subtract_value, dtype=torch.float16, device=x.device)
        x = kernel(x, subtract_val_tensor)
        kernel = self._get_maxpool_kernel(B, C, H, W, "float16")
        x = kernel(x)
        B, C, H, W = x.shape
        kernel = self._get_mish_kernel(B, C, H, W, "float16")
        x = kernel(x)
        return x