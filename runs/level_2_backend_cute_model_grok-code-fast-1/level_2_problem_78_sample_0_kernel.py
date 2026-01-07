import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def sum_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    thread_idx = bidx * bdim + tidx

    batch, channels, depth, height, width = gX.shape
    total_spatial = depth * height * width
    total_output = batch * total_spatial

    if thread_idx >= total_output:
        return

    bi = thread_idx // total_spatial
    spatial_idx = thread_idx % total_spatial
    di = spatial_idx // (height * width)
    hw = spatial_idx % (height * width)
    hi = hw // width
    wi = hw % width

    sum_val = 0.0
    for ci in range(channels):
        sum_val += gX[bi, ci, di, hi, wi]

    gY[bi, 0, di, hi, wi] = sum_val

@cute.jit
def sum_host(mX: cute.Tensor, mY: cute.Tensor):
    batch, channels, depth, height, width = mX.shape
    total_output = batch * depth * height * width

    threads_per_block = 256
    grid_x = cute.ceil_div(total_output, threads_per_block)

    sum_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a custom sum operation using CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        # Custom sum over dim=1 using CuTe kernel
        B, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        y = torch.empty((B, 1, D, H, W), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(sum_host, mX, mY)
            self.compiled[key] = compiled

        compiled(mX, mY)
        return y