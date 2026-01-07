import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_tanh_scale_bias_kernel(gA: cute.Tensor, scale: float, gBias: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    B, C, H, W = gA.shape
    total_elems = B * C * H * W
    if thread_idx >= total_elems:
        return

    bi = thread_idx // (C * H * W)
    ci = (thread_idx % (C * H * W)) // (H * W)
    hi = (thread_idx % (H * W)) // W
    wi = thread_idx % W

    a_val = gA[bi, ci, hi, wi]
    bias_val = gBias[ci, 0, 0]  # bias is (C, 1, 1)

    gC[bi, ci, hi, wi] = torch.tanh(a_val) * scale + bias_val

@cute.jit
def fused_tanh_scale_bias_host(mA: cute.Tensor, scale: float, mBias: cute.Tensor, mC: cute.Tensor):
    B, C, H, W = mA.shape
    total_elems = B * C * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_tanh_scale_bias_kernel(mA, scale, mBias, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model with custom CuTe kernel for fused tanh, scaling, and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.compiled = {}

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Fused tanh, scaling, bias addition using custom kernel
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty_like(x)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mC = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_tanh_scale_bias_host, mA, self.scaling_factor, mBias, mC)
            self.compiled[key] = compiled

        compiled(mA, self.scaling_factor, mBias, mC)
        # Max-pooling
        out = self.max_pool(out)
        return out