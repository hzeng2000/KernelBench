import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_div_leaky_kernel(gA: cute.Tensor, gC: cute.Tensor, divisor: float): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    total_elems = gA.numel()
    if thread_idx < total_elems:
        a_val = gA.flat[thread_idx]
        val = a_val / divisor
        gC.flat[thread_idx] = val if val > 0.0 else 0.01 * val

@cute.jit
def fused_div_leaky_host(mA: cute.Tensor, mC: cute.Tensor, divisor: float):
    total_elems = mA.numel()

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_div_leaky_kernel(mA, mC, divisor).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, then fuses divide by constant and LeakyReLU into a single custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        # Fuse divide and leaky_relu into custom kernel
        x = x.contiguous().cuda()
        out = torch.empty_like(x)
        
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype, self.divisor)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_div_leaky_host, mA, mC, self.divisor)
            self.compiled[key] = compiled
        
        compiled(mA, mC, self.divisor)
        return out