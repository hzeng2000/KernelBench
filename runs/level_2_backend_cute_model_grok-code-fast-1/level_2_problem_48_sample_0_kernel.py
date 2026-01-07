import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_post_conv_kernel(gX: cute.Tensor, gScaling: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    B, C, D, H, W = gX.shape
    total_elems = B * C * D * H * W

    if thread_idx >= total_elems:
        return

    idx = thread_idx
    b = idx // (C * D * H * W)
    idx %= (C * D * H * W)
    c = idx // (D * H * W)
    idx %= (D * H * W)
    d = idx // (H * W)
    idx %= (H * W)
    h = idx // W
    w = idx % W

    x_val = gX[b, c, d, h, w]
    scaling_val = gScaling[c, 0, 0, 0]
    bias_val = gBias[c, 0, 0, 0]

    temp = x_val * scaling_val
    temp = math.tanh(temp)
    temp = temp * bias_val
    temp = 1.0 / (1.0 + math.exp(-temp))

    gOut[b, c, d, h, w] = temp

@cute.jit
def fused_post_conv_host(mX: cute.Tensor, mScaling: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor):
    B, C, D, H, W = mX.shape
    total_elems = B * C * D * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_post_conv_kernel(mX, mScaling, mBias, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, then fuses the scaling, tanh, bias multiplication, and sigmoid into a single CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        scaling = self.scaling_factor.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty((B, C, D, H, W), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mScaling = from_dlpack(scaling, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_conv_host, mX, mScaling, mBias, mOut)
            self.compiled[key] = compiled

        compiled(mX, mScaling, mBias, mOut)
        return out