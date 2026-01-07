import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv2d_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    N: int, C: int, H: int, W: int, K: int, R: int, S: int, P: int, Q: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    # Output position
    n = bidx
    k = bidy * bdimy + tidy
    p = bidz * bdimz + tidz

    if n < N and k < K and p < P:
        for q in range(Q):
            acc = 0.0
            for c in range(C):
                for r in range(R):
                    for s in range(S):
                        h_in = p + r
                        w_in = q + s
                        if h_in < H and w_in < W:
                            acc += gX[n, c, h_in, w_in] * gW[k, c, r, s]
            if k < K and p < P and q < Q:
                gY[n, k, p, q] = acc + gB[k]

@cute.kernel
def instance_norm_kernel(
    gX: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor, gY: cute.Tensor,
    N: int, C: int, H: int, W: int, eps: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    n = bidx
    c = bidy * bdimy + tidy

    if n < N and c < C:
        # Compute mean
        sum_val = 0.0
        for h in range(H):
            for w in range(W):
                sum_val += gX[n, c, h, w]
        mean = sum_val / (H * W)
        gMean[n, c] = mean

        # Compute variance
        var_sum = 0.0
        for h in range(H):
            for w in range(W):
                diff = gX[n, c, h, w] - mean
                var_sum += diff * diff
        var = var_sum / (H * W)
        gVar[n, c] = var

        # Normalize
        inv_std = 1.0 / (var + eps)**0.5
        for h in range(H):
            for w in range(W):
                gY[n, c, h, w] = (gX[n, c, h, w] - mean) * inv_std

@cute.kernel
def divide_kernel(
    gX: cute.Tensor, gY: cute.Tensor, scalar: float,
    N: int, C: int, H: int, W: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    n = bidx
    c = bidy * bdimy + tidy
    h = bidz * bdimz + tidz

    if n < N and c < C and h < H:
        for w in range(W):
            gY[n, c, h, w] = gX[n, c, h, w] / scalar

@cute.jit
def fused_conv_norm_div_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    divide_by: float, eps: float
):
    N, C, H, W = mX.shape
    K, _, R, S = mW.shape
    P = H - R + 1
    Q = W - S + 1

    # Conv2D
    threads_per_block = (8, 8, 8)
    grid = (N, (K + 7) // 8, (P + 7) // 8)
    conv2d_kernel(mX, mW, mB, mY, N, C, H, W, K, R, S, P, Q).launch(
        grid=grid, block=threads_per_block
    )

    # Instance Norm
    mean = torch.zeros(N, K, device=mX.device, dtype=mX.dtype)
    var = torch.zeros(N, K, device=mX.device, dtype=mX.dtype)
    mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    mY_re = from_dlpack(mY, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

    grid_norm = (N, (K + 7) // 8, 1)
    threads_norm = (1, 8, 1)
    instance_norm_kernel(mY_re, mMean, mVar, mY_re, N, K, P, Q, eps).launch(
        grid=grid_norm, block=threads_norm
    )

    # Divide
    grid_div = (N, (K + 7) // 8, (P + 7) // 8)
    threads_div = (1, 8, 8)
    divide_kernel(mY_re, mY_re, divide_by, N, K, P, Q).launch(
        grid=grid_div, block=threads_div
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.divide_by = divide_by
        self.eps = 1e-5
        self.compiled = None

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.weight.shape[0]
        P = H - self.weight.shape[2] + 1
        Q = W - self.weight.shape[3] + 1

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty(N, K, P, Q, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        if self.compiled is None:
            self.compiled = cute.compile(
                fused_conv_norm_div_host, mX, mW, mB, mY, self.divide_by, self.eps
            )

        self.compiled(mX, mW, mB, mY, self.divide_by, self.eps)
        return out