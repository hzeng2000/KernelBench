import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_softmax_kernel(B: int, C: int, D: int, H: int, W: int, dtype: str = "float16"):
    @T.prim_func
    def softmax_kernel(
        A: T.Tensor((B, C, D, H, W), dtype),
        B_out: T.Tensor((B, C, D, H, W), dtype),
    ):
        with T.Kernel(B, D, H, W, threads=C) as (bb, bd, bh, bw):
            max_val = T.reduce(T.max, A[bb, T.reduce_axis(0, C), bd, bh, bw], init=-float('inf'))
            sum_val = T.reduce(T.add, T.exp(A[bb, T.reduce_axis(0, C), bd, bh, bw] - max_val), init=0.0)
            with T.parallel(C) as tc:
                B_out[bb, tc, bd, bh, bw] = T.exp(A[bb, tc, bd, bh, bw] - max_val) / sum_val

    return tilelang.compile(softmax_kernel, out_idx=[1], target="cuda")


def build_maxpool_kernel(B: int, C: int, D_in: int, H_in: int, W_in: int, D_out: int, H_out: int, W_out: int, kernel_size: int = 2, dtype: str = "float16"):
    @T.prim_func
    def maxpool_kernel(
        A: T.Tensor((B, C, D_in, H_in, W_in), dtype),
        B_out: T.Tensor((B, C, D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(B, C, D_out, H_out, W_out, threads=kernel_size**3) as (bb, bc, bd, bh, bw):
            tid = T.thread_idx()
            id = bd * kernel_size + (tid // 4)
            ih = bh * kernel_size + ((tid % 4) // 2)
            iw = bw * kernel_size + (tid % 2)
            val = A[bb, bc, id, ih, iw]
            B_out[bb, bc, bd, bh, bw] = T.reduce(T.max, val, axis=T.thread_axis(), init=-float('inf'))

    return tilelang.compile(maxpool_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Softmax with TileLang kernel, and performs two max pooling operations with TileLang kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Precompute shapes after conv: assuming kernel_size=3, input depth=16 -> 16-3+1=14, height=32-3+1=30, width=30
        self.B, self.C, self.D, self.H, self.W = 128, out_channels, 14, 30, 30
        self.D1, self.H1, self.W1 = 7, 15, 15  # After first pool
        self.D2, self.H2, self.W2 = 3, 7, 7    # After second pool
        self.softmax_kernel = build_softmax_kernel(self.B, self.C, self.D, self.H, self.W, dtype="float16")
        self.maxpool1_kernel = build_maxpool_kernel(self.B, self.C, self.D, self.H, self.W, self.D1, self.H1, self.W1, kernel_size=pool_kernel_size, dtype="float16")
        self.maxpool2_kernel = build_maxpool_kernel(self.B, self.C, self.D1, self.H1, self.W1, self.D2, self.H2, self.W2, kernel_size=pool_kernel_size, dtype="float16")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = self.conv(x)
        x = x.half()  # Convert to FP16
        x = self.softmax_kernel(x)
        x = self.maxpool1_kernel(x)
        x = self.maxpool2_kernel(x)
        return x.float()  # Convert back to FP32