import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_bias_scale_sigmoid_kernel(gA: cute.Tensor, gBias: cute.Tensor, gScale: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    B, C, H, W = gA.shape
    total_per_batch = C * H * W
    bi = thread_idx // total_per_batch
    remaining = thread_idx % total_per_batch
    ci = remaining // (H * W)
    remaining2 = remaining % (H * W)
    hi = remaining2 // W
    wi = remaining2 % W

    if bi < B and ci < C and hi < H and wi < W:
        a_val = gA[bi, ci, hi, wi]
        bias_val = gBias[ci, 0, 0]
        scale_val = gScale[ci, 0, 0]
        scaled = (a_val + bias_val) * scale_val
        gC[bi, ci, hi, wi] = 1.0 / (1.0 + math.exp(-scaled))

@cute.jit
def fused_bias_scale_sigmoid_host(mA: cute.Tensor, mBias: cute.Tensor, mScale: cute.Tensor, mC: cute.Tensor):
    B, C, H, W = mA.shape
    total_elems = B * C * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_bias_scale_sigmoid_kernel(mA, mBias, mScale, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, fused bias add, scale multiply, and sigmoid using custom CuTe kernel, followed by group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        scale = self.scale.contiguous().cuda()
        out = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mScale = from_dlpack(scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mC = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_bias_scale_sigmoid_host, mA, mBias, mScale, mC)
            self.compiled[key] = compiled

        compiled(mA, mBias, mScale, mC)
        x = self.group_norm(out)
        return x