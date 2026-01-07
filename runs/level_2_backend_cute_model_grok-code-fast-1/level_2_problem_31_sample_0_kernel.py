import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_post_conv_kernel(gX: cute.Tensor, gBias: cute.Tensor, constant: float, scaling: float, gY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    B, C, H, W = gX.shape
    total_elems = B * C * H * W

    if thread_idx >= total_elems:
        return

    w_idx = thread_idx % W
    h_idx = (thread_idx // W) % H
    c_idx = ((thread_idx // W) // H) % C
    b_idx = (((thread_idx // W) // H) // C) % B

    val = gX[b_idx, c_idx, h_idx, w_idx]
    val = min(val, constant)
    val = val + gBias[c_idx, 0, 0]
    val = val * scaling
    gY[b_idx, c_idx, h_idx, w_idx] = val

@cute.jit
def fused_post_conv_host(mX: cute.Tensor, mBias: cute.Tensor, constant: float, scaling: float, mY: cute.Tensor):
    B, C, H, W = mX.shape
    total_elems = B * C * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_post_conv_kernel(mX, mBias, constant, scaling, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model with fused post-conv operations using custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        y = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_conv_host, mX, mBias, self.constant_value, self.scaling_factor, mY)
            self.compiled[key] = compiled

        compiled(mX, mBias, self.constant_value, self.scaling_factor, mY)
        return y