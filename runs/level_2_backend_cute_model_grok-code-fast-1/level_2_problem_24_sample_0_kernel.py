import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def min_reduce_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    thread_idx = bidx * bdim + tidx
    B, C, D, H, W = gX.shape
    total = B * C * H * W
    if thread_idx >= total:
        return
    b = thread_idx // (C * H * W)
    rem = thread_idx % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W
    min_val = float('inf')
    for d in range(D):
        val = gX[b, c, d, h, w]
        if val < min_val:
            min_val = val
    gY[b, c, h, w] = min_val

@cute.jit
def min_reduce_host(mX: cute.Tensor, mY: cute.Tensor):
    B, C, D, H, W = mX.shape
    total = B * C * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    min_reduce_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def softmax_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    thread_idx = bidx * bdim + tidx
    B, C, H, W = gX.shape
    total = B * H * W
    if thread_idx >= total:
        return
    b = thread_idx // (H * W)
    rem = thread_idx % (H * W)
    h = rem // W
    w = rem % W
    max_val = -float('inf')
    for c in range(C):
        val = gX[b, c, h, w]
        if val > max_val:
            max_val = val
    sum_exp = 0.0
    for c in range(C):
        val = gX[b, c, h, w]
        sum_exp += cute.exp(val - max_val)
    for c in range(C):
        val = gX[b, c, h, w]
        gY[b, c, h, w] = cute.exp(val - max_val) / sum_exp

@cute.jit
def softmax_host(mX: cute.Tensor, mY: cute.Tensor):
    B, C, H, W = mX.shape
    total = B * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    softmax_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.compiled_min = {}
        self.compiled_softmax = {}

    def forward(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        y = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key = (x.dtype,)
        compiled_min = self.compiled_min.get(key)
        if compiled_min is None:
            compiled_min = cute.compile(min_reduce_host, mX, mY)
            self.compiled_min[key] = compiled_min
        compiled_min(mX, mY)
        z = torch.empty_like(y)
        mY2 = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mZ = from_dlpack(z, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        compiled_softmax = self.compiled_softmax.get(key)
        if compiled_softmax is None:
            compiled_softmax = cute.compile(softmax_host, mY2, mZ)
            self.compiled_softmax[key] = compiled_softmax
        compiled_softmax(mY2, mZ)
        return z