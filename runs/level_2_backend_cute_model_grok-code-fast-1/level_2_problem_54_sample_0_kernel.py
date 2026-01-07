import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_kernel(gX: cute.Tensor, gMultiplier: cute.Tensor, gOut: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    batch, out, h, w = gX.shape
    total_elems = batch * out * h * w

    if thread_idx >= total_elems:
        return

    bi = thread_idx // (out * h * w)
    temp = thread_idx % (out * h * w)
    ci = temp // (h * w)
    temp2 = temp % (h * w)
    hi = temp2 // w
    wi = temp2 % w

    x_val = gX[bi, ci, hi, wi]
    mult_val = gMultiplier[ci, 0, 0]
    val = x_val * mult_val

    # LeakyReLU
    val = val if val > 0.0 else 0.01 * val

    # GELU approximation using erf (assuming CUDA erff is available)
    import math
    val = 0.5 * val * (1.0 + math.erf(val / math.sqrt(2.0)))

    gOut[bi, ci, hi, wi] = val

@cute.jit
def fused_host(mX: cute.Tensor, mMultiplier: cute.Tensor, mOut: cute.Tensor):
    batch, out, h, w = mX.shape
    total_elems = batch * out * h * w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_kernel(mX, mMultiplier, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        batch, out, h, w = x.shape
        x = x.contiguous().cuda()
        multiplier = self.multiplier.contiguous().cuda()
        out_tensor = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mMultiplier = from_dlpack(multiplier, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mOut = from_dlpack(out_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_host, mX, mMultiplier, mOut)
            self.compiled[key] = compiled

        compiled(mX, mMultiplier, mOut)
        return out_tensor