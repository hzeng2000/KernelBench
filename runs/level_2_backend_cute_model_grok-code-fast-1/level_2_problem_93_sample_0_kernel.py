import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def post_conv_kernel(gX: cute.Tensor, add_val: float, mul_val: float, gOut: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    total_elems = gX.size()
    if thread_idx >= total_elems:
        return

    x_val = gX.flat[thread_idx]
    x_val = x_val + add_val
    x_val = min(x_val, 0.0)
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt2 = math.sqrt(2.0)
    erf_val = math.erf(x_val / sqrt2)
    gelu_val = 0.5 * x_val * (1.0 + erf_val)
    x_val = gelu_val
    x_val = x_val * mul_val
    gOut.flat[thread_idx] = x_val

@cute.jit
def post_conv_host(mX: cute.Tensor, add_val: float, mul_val: float, mOut: cute.Tensor):
    total_elems = mX.size()
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    post_conv_kernel(mX, add_val, mul_val, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then fuses add, min, GELU, and multiply into a single custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous().cuda()
        out = torch.empty_like(x)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(post_conv_host, mX, self.add_value, self.multiply_value, mOut)
            self.compiled[key] = compiled
        
        compiled(mX, self.add_value, self.multiply_value, mOut)
        return out