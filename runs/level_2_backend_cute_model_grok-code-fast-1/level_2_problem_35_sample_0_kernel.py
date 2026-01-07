import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def subtract_hardswish_kernel(gX: cute.Tensor, subtract_value: float, gY: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    total_elems = gX.size()
    if thread_idx >= total_elems:
        return

    val = gX.flat[thread_idx] - subtract_value
    clamped = cute.max(cute.min(val + 3.0, 6.0), 0.0)
    gY.flat[thread_idx] = val * clamped / 6.0

@cute.jit
def subtract_hardswish_host(mX: cute.Tensor, subtract_value: float, mY: cute.Tensor):
    total_elems = mX.size()

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    subtract_hardswish_kernel(mX, subtract_value, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def mish_kernel(gX: cute.Tensor, gY: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    total_elems = gX.size()
    if thread_idx >= total_elems:
        return

    val = gX.flat[thread_idx]
    softplus = cute.log(1.0 + cute.exp(val))
    gY.flat[thread_idx] = val * cute.tanh(softplus)

@cute.jit
def mish_host(mX: cute.Tensor, mY: cute.Tensor):
    total_elems = mX.size()

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    mish_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.subtract_hardswish_compiled = {}
        self.mish_compiled = {}

    def forward(self, x):
        x = self.conv(x)
        # Custom subtract + hardswish
        x_shape = x.shape
        x = x.contiguous().cuda()
        y = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key = (x.dtype,)
        compiled = self.subtract_hardswish_compiled.get(key)
        if compiled is None:
            compiled = cute.compile(subtract_hardswish_host, mX, self.subtract_value, mY)
            self.subtract_hardswish_compiled[key] = compiled
        compiled(mX, self.subtract_value, mY)
        x = y
        x = self.pool(x)
        # Custom mish
        x = x.contiguous().cuda()
        y = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key = (x.dtype,)
        compiled = self.mish_compiled.get(key)
        if compiled is None:
            compiled = cute.compile(mish_host, mX, mY)
            self.mish_compiled[key] = compiled
        compiled(mX, mY)
        x = y
        return x