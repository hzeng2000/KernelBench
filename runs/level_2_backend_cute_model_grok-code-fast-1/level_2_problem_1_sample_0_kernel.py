import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def relu_bias_kernel(gX: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor): 
    tidx = cute.arch.thread_idx(0)  
    bidx = cute.arch.block_idx(0)  
    bdim = cute.arch.block_dim(0)  

    thread_idx = bidx * bdim + tidx

    B, C, H, W = gX.shape
    total = B * C * H * W

    if thread_idx >= total:
        return

    bi = thread_idx // (C * H * W)
    ci = (thread_idx % (C * H * W)) // (H * W)
    hi = (thread_idx % (H * W)) // W
    wi = thread_idx % W

    x_val = gX[bi, ci, hi, wi]
    b_val = gBias[ci, 0, 0]

    gOut[bi, ci, hi, wi] = max(0.0, x_val + b_val)

@cute.jit
def relu_bias_host(mX: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor):
    B, C, H, W = mX.shape
    total = B * C * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)

    relu_bias_kernel(mX, mBias, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model that performs convolution, then fused ReLU and bias addition using a custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(relu_bias_host, mX, mBias, mOut)
            self.compiled[key] = compiled

        compiled(mX, mBias, mOut)
        return out