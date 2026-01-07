import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_sub_mish_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    subtract1: float, subtract2: float,
    batch_size: int, in_channels: int, out_channels: int,
    height: int, width: int, kernel_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    gdimx, gdimy, gdimz = cute.arch.grid_dim()

    out_h = height - kernel_size + 1
    out_w = width - kernel_size + 1

    hw = out_h * out_w
    hw_total = batch_size * out_channels * hw

    tid = bidz * gdimx * gdimy * bdimx * bdimy + \
          bidy * gdimx * bdimx * bdimy + \
          bidx * bdimx * bdimy + \
          tidy * bdimx + tidx

    if tid >= hw_total:
        return

    n = tid // (out_channels * hw)
    c_out = (tid // hw) % out_channels
    hw_idx = tid % hw
    h_out = hw_idx // out_w
    w_out = hw_idx % out_w

    acc = 0.0
    if gB is not None:
        acc = gB[c_out]

    for c_in in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                h_in = h_out + kh
                w_in = w_out + kw
                x_val = gX[n, c_in, h_in, w_in]
                w_val = gW[c_out, c_in, kh, kw]
                acc += x_val * w_val

    acc = acc - subtract1
    acc = acc - subtract2

    # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    exp_val = cute.math.exp(acc)
    softplus = cute.math.log(1.0 + exp_val)
    tanh_val = cute.math.tanh(softplus)
    mish_val = acc * tanh_val

    gY[n, c_out, h_out, w_out] = mish_val

@cute.jit
def conv_sub_mish_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    subtract1: float, subtract2: float,
    batch_size: int, in_channels: int, out_channels: int,
    height: int, width: int, kernel_size: int
):
    out_h = height - kernel_size + 1
    out_w = width - kernel_size + 1
    total_threads = batch_size * out_channels * out_h * out_w

    threads_per_block = 256
    grid_x = cute.ceil_div(total_threads, threads_per_block)

    conv_sub_mish_kernel(
        mX, mW, mB, mY, subtract1, subtract2,
        batch_size, in_channels, out_channels, height, width, kernel_size
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x = x.contiguous().cuda()
        weight = self.conv.weight
        bias = self.conv.bias
        out_h = height - self.conv.kernel_size[0] + 1
        out_w = width - self.conv.kernel_size[1] + 1
        out_channels = self.conv.out_channels
        in_channels = self.conv.in_channels
        kernel_size = self.conv.kernel_size[0]

        y = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_sub_mish_host,
                mX, mW, mB, mY,
                self.subtract_value_1, self.subtract_value_2,
                batch_size, in_channels, out_channels, height, width, kernel_size
            )
            self.compiled[key] = compiled

        compiled(
            mX, mW, mB, mY,
            self.subtract_value_1, self.subtract_value_2,
            batch_size, in_channels, out_channels, height, width, kernel_size
        )
        return y