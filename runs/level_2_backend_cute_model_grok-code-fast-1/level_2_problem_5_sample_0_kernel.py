import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def bias_tanh_kernel(gX: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor): 
    tidx = cute.arch.thread_idx().x  
    bidx = cute.arch.block_idx().x  
    bdim = cute.arch.block_dim().x  

    thread_idx = bidx * bdim + tidx

    B, C, H, W = gX.shape
    total_elems = B * C * H * W

    if thread_idx >= total_elems:
        return

    w = thread_idx % W
    h = (thread_idx // W) % H
    c = (thread_idx // (W * H)) % C
    b = thread_idx // (W * H * C)

    x_val = gX[b, c, h, w]
    bias_val = gBias[c, 0, 0]
    gOut[b, c, h, w] = cute.tanh(x_val - bias_val)

@cute.jit
def bias_tanh_host(mX: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor):
    B, C, H, W = mX.shape
    total_elems = B * C * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    bias_tanh_kernel(mX, mBias, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then fuses bias subtraction and tanh into a single CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(bias_tanh_host, mX, mBias, mOut)
            self.compiled[key] = compiled

        compiled(mX, mBias, mOut)
        return out