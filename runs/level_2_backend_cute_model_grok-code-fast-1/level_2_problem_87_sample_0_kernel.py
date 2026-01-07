import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def mish_kernel(gX: cute.Tensor, sub1: float, sub2: float, gOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    B, C, H, W = gX.shape
    total_elems = B * C * H * W
    if thread_idx >= total_elems:
        return

    stride_B = C * H * W
    stride_C = H * W
    stride_H = W

    b = thread_idx // stride_B
    rem = thread_idx % stride_B
    c = rem // stride_C
    rem2 = rem % stride_C
    h = rem2 // stride_H
    w = rem2 % stride_H

    val = gX[b, c, h, w] - sub1 - sub2
    exp_val = cute.exp(val)
    softplus = cute.log(1.0 + exp_val)
    tanh_softplus = cute.tanh(softplus)
    mish_val = val * tanh_softplus
    gOut[b, c, h, w] = mish_val

@cute.jit
def mish_host(mX: cute.Tensor, sub1: float, sub2: float, mOut: cute.Tensor):
    B, C, H, W = mX.shape
    total_elems = B * C * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    mish_kernel(mX, sub1, sub2, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().cuda()
        out = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(mish_host, mX, self.subtract_value_1, self.subtract_value_2, mOut)
            self.compiled[key] = compiled

        compiled(mX, self.subtract_value_1, self.subtract_value_2, mOut)
        return out