import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_maxpool_hardtanh_kernel(B: int, C: int, H: int, W: int, Hp: int, Wp: int, kernel_size: int, stride: int, min_val: float, max_val: float, block_B: int = 1, block_C: int = 1, block_H: int = 1, block_W: int = 1, threads: int = 128):
    
    @T.prim_func
    def maxpool_hardtanh_kernel(
        A: T.Tensor((B, C, H, W), "float16"),
        B_out: T.Tensor((B, C, Hp, Wp), "float16"),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(C, block_C), T.ceildiv(Hp, block_H), T.ceildiv(Wp, block_W), threads=threads) as (bx, by, bz, bw):
            for lb, lc, lh, lw in T.Parallel(block_B, block_C, block_H, block_W):
                b = bx * block_B + lb
                c = by * block_C + lc
                hp = bz * block_H + lh
                wp = bw * block_W + lw

                if b < B and c < C and hp < Hp and wp < Wp:
                    max_val_frag = T.alloc_fragment((1,), "float16")
                    T.fill(max_val_frag, -float('inf'))
                    for kh in T.serial(kernel_size):
                        for kw in T.serial(kernel_size):
                            val = A[b, c, hp * stride + kh, wp * stride + kw]
                            max_val_frag[0] = T.max(max_val_frag[0], val)
                    B_out[b, c, hp, wp] = T.clamp(max_val_frag[0], min_val, max_val)

    return tilelang.compile(maxpool_hardtanh_kernel, out_idx=[1], target="cuda")


def build_mean_tanh_kernel(B: int, C: int, Hp: int, Wp: int, block_B: int = 1, block_C: int = 1, threads: int = 128):
    
    @T.prim_func
    def mean_tanh_kernel(
        B_in: T.Tensor((B, C, Hp, Wp), "float16"),
        C_out: T.Tensor((B, C, 1, 1), "float16"),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(C, block_C), threads=threads) as (bx, by):
            for lb, lc in T.Parallel(block_B, block_C):
                b = bx * block_B + lb
                c = by * block_C + lc

                if b < B and c < C:
                    sum_val = T.alloc_fragment((1,), "float32")
                    T.clear(sum_val)
                    for hp in T.serial(Hp):
                        for wp in T.serial(Wp):
                            sum_val[0] += B_in[b, c, hp, wp]
                    mean_val = sum_val[0] / (Hp * Wp)
                    C_out[b, c, 0, 0] = T.tanh(mean_val)

    return tilelang.compile(mean_tanh_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, followed by fused maxpool+hardtanh, then fused mean+tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self._kernel_cache = {}

    def _get_maxpool_kernel(self, B: int, C: int, H: int, W: int, Hp: int, Wp: int):
        key = ("maxpool", B, C, H, W, Hp, Wp)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_maxpool_hardtanh_kernel(B, C, H, W, Hp, Wp, self.maxpool_kernel_size, self.maxpool_stride, self.hardtanh_min, self.hardtanh_max)
        return self._kernel_cache[key]

    def _get_mean_kernel(self, B: int, C: int, Hp: int, Wp: int):
        key = ("mean", B, C, Hp, Wp)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_mean_tanh_kernel(B, C, Hp, Wp)
        return self._kernel_cache[key]

    def forward(self, x):
        x = self.conv_transpose(x.half())  # Ensure FP16
        B, C, H, W = x.shape
        Hp = H // self.maxpool_stride
        Wp = W // self.maxpool_stride
        maxpool_kernel = self._get_maxpool_kernel(B, C, H, W, Hp, Wp)
        pooled = maxpool_kernel(x)
        mean_kernel = self._get_mean_kernel(B, C, Hp, Wp)
        out = mean_kernel(pooled)
        return out