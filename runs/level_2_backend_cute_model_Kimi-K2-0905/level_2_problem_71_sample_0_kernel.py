import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_div_leakyrelu_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    N: int, H: int, W: int, C_in: int, C_out: int,
    kernel_size: int, divisor: float, negative_slope: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    gdimx, gdimy, gdimz = cute.arch.grid_dim()

    hw = bidx * bdimx + tidx
    c_out = bidy * bdimy + tidy
    n = bidz * bdimz + tidz

    if n >= N or c_out >= C_out or hw >= (H * W):
        return

    h = hw // W
    w = hw % W
    kh = kernel_size // 2

    acc = 0.0
    for c_in in range(C_in):
        for kh_idx in range(kernel_size):
            for kw_idx in range(kernel_size):
                h_in = h - kh + kh_idx
                w_in = w - kh + kw_idx
                if 0 <= h_in < H and 0 <= w_in < W:
                    x_val = gX[n, c_in, h_in, w_in]
                    w_val = gW[c_out, c_in, kh_idx, kw_idx]
                    acc += x_val * w_val

    if gB is not None:
        acc += gB[c_out]

    acc = acc / divisor
    if acc < 0.0:
        acc = acc * negative_slope
    gY[n, c_out, h, w] = acc

@cute.jit
def conv_div_leakyrelu_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    N: int, H: int, W: int, C_in: int, C_out: int,
    kernel_size: int, divisor: float, negative_slope: float
):
    total_hw = H * W
    threads_x = 16
    threads_y = 16
    threads_z = 4
    blocks_x = cute.ceil_div(total_hw, threads_x)
    blocks_y = cute.ceil_div(C_out, threads_y)
    blocks_z = cute.ceil_div(N, threads_z)

    conv_div_leakyrelu_kernel(
        mX, mW, mB, mY, N, H, W, C_in, C_out,
        kernel_size, divisor, negative_slope
    ).launch(
        grid=(blocks_x, blocks_y, blocks_z),
        block=(threads_x, threads_y, threads_z)
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.negative_slope = 0.01
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H, W = x.shape
        C_out = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        x = x.contiguous().cuda()
        weight = self.conv.weight.data.contiguous().cuda()
        bias = self.conv.bias.data.contiguous().cuda() if self.conv.bias is not None else None

        y = torch.empty(N, C_out, H, W, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,)) if bias is not None else None
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_div_leakyrelu_host,
                mX, mW, mB, mY,
                N, H, W, C_in, C_out,
                kernel_size, self.divisor, self.negative_slope
            )
            self.compiled[key] = compiled

        compiled(
            mX, mW, mB, mY,
            N, H, W, C_in, C_out,
            kernel_size, self.divisor, self.negative_slope
        )
        return y