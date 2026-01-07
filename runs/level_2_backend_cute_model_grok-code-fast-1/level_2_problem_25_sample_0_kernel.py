import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def min_tanh_tanh_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    thread_idx = bidx * bdim + tidx

    N, C, H, W = gX.shape
    total_spatial = N * H * W
    if thread_idx >= total_spatial:
        return

    n = thread_idx // (H * W)
    hw = thread_idx % (H * W)
    h = hw // W
    w = hw % W

    min_val = float('inf')
    for c in range(C):
        val = gX[n, c, h, w]
        min_val = min(min_val, val)

    # Compute tanh(tanh(min_val)) using the formula
    def tanh(x):
        exp2x = cute.exp(2 * x)
        return (exp2x - 1) / (exp2x + 1)

    temp = tanh(min_val)
    result = tanh(temp)
    gY[n, 0, h, w] = result

@cute.jit
def min_tanh_tanh_host(mX: cute.Tensor, mY: cute.Tensor):
    N, C, H, W = mX.shape
    total_threads = N * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total_threads, threads_per_block)
    min_tanh_tanh_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = cutlass.Conv2d(in_channels, out_channels, kernel_size, kernel_size, stride=1, padding=0, dtype=torch.float32)
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        x = x.contiguous().cuda()
        y = torch.empty((N, 1, H, W), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype, C, H, W)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(min_tanh_tanh_host, mX, mY)
            self.compiled[key] = compiled

        compiled(mX, mY)
        return y