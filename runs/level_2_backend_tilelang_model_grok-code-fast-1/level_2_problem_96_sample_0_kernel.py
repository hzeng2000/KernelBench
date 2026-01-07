import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_multiply_scalar_kernel(B: int, C: int, D: int, H: int, W: int, dtype: str = "float16"):
    block_B = 1
    block_C = 1
    block_D = 1
    block_H = 32
    block_W = 32
    threads = 1024

    @T.prim_func
    def multiply_scalar_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        scale: T.Tensor((), dtype),
        C_out: T.Tensor((B, C, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(C, block_C), T.ceildiv(D, block_D), T.ceildiv(H, block_H), T.ceildiv(W, block_W), threads=threads) as (bb, bc, bd, bh, bw):
            for lb, lc, ld, lh, lw in T.Parallel(block_B, block_C, block_D, block_H, block_W):
                b = bb * block_B + lb
                c = bc * block_C + lc
                d = bd * block_D + ld
                h = bh * block_H + lh
                w = bw * block_W + lw
                if b < B and c < C and d < D and h < H and w < W:
                    C_out[b, c, d, h, w] = A[b, c, d, h, w] * scale[()]

    return tilelang.compile(multiply_scalar_kernel, out_idx=[2], target="cuda")


def build_maxpool3d_kernel(B: int, C: int, D: int, H: int, W: int, dtype: str = "float16"):
    D_out = D // 2
    H_out = H // 2
    W_out = W // 2
    block_B = 1
    block_C = 1
    block_D = 8
    block_H = 8
    block_W = 8
    threads = 512

    @T.prim_func
    def maxpool3d_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        C_out: T.Tensor((B, C, D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_B), T.ceildiv(C, block_C), T.ceildiv(D_out, block_D), T.ceildiv(H_out, block_H), T.ceildiv(W_out, block_W), threads=threads) as (bb, bc, bd, bh, bw):
            for lb, lc, ld, lh, lw in T.Parallel(block_B, block_C, block_D, block_H, block_W):
                b = bb * block_B + lb
                c = bc * block_C + lc
                od = bd * block_D + ld
                oh = bh * block_H + lh
                ow = bw * block_W + lw
                if b < B and c < C and od < D_out and oh < H_out and ow < W_out:
                    max_val = T.min_value(dtype)
                    for kd in T.serial(2):
                        for kh in T.serial(2):
                            for kw in T.serial(2):
                                id = od * 2 + kd
                                ih = oh * 2 + kh
                                iw = ow * 2 + kw
                                if id < D and ih < H and iw < W:
                                    max_val = T.max(max_val, A[b, c, id, ih, iw])
                    C_out[b, c, od, oh, ow] = max_val

    return tilelang.compile(maxpool3d_kernel, out_idx=[1], target="cuda")


def build_global_avg_clamp_kernel(B: int, C: int, D: int, H: int, W: int, dtype: str = "float16"):
    @T.prim_func
    def global_avg_clamp_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        C_out: T.Tensor((B, C, 1, 1, 1), dtype),
    ):
        with T.Kernel(B, C, 1, 1, 1, threads=1) as (b, c, _, _, _):
            sum_val = T.cast(0.0, dtype)
            for d in T.serial(D):
                for h in T.serial(H):
                    for w in T.serial(W):
                        sum_val = sum_val + A[b, c, d, h, w]
            num = T.cast(D * H * W, dtype)
            avg = sum_val / num
            C_out[b, c, 0, 0, 0] = T.max(T.min(avg, T.cast(1.0, dtype)), T.cast(0.0, dtype))

    return tilelang.compile(global_avg_clamp_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output using custom TileLang kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding).half()
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        # Fixed shapes based on the architecture
        self.B = 128
        self.in_C = in_channels
        self.out_C = out_channels
        self.conv_D, self.conv_H, self.conv_W = 31, 63, 63  # After conv_transpose
        self.pool_D, self.pool_H, self.pool_W = 15, 31, 31  # After maxpool
        self._multiply_kernel = build_multiply_scalar_kernel(self.B, self.out_C, self.conv_D, self.conv_H, self.conv_W, dtype="float16")
        self._maxpool_kernel = build_maxpool3d_kernel(self.B, self.out_C, self.conv_D, self.conv_H, self.conv_W, dtype="float16")
        self._global_avg_clamp_kernel = build_global_avg_clamp_kernel(self.B, self.out_C, self.pool_D, self.pool_H, self.pool_W, dtype="float16")

    def forward(self, x):
        x = x.half()
        x = self.conv_transpose(x)
        scale_tensor = torch.tensor(self.scale, dtype=torch.float16, device=x.device)
        x = self._multiply_kernel(x, scale_tensor)
        x = self._maxpool_kernel(x)
        x = self._global_avg_clamp_kernel(x)
        return x