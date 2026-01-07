import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_activation_kernel(gA: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    total_elems = gA.numel()
    if thread_idx < total_elems:
        a_val = gA.flat[thread_idx]
        relu_val = max(0.0, a_val)
        clamp_val = min(1.0, max(0.0, (relu_val + 3.0) / 6.0))
        gC.flat[thread_idx] = relu_val * clamp_val

@cute.jit
def fused_activation_host(mA: cute.Tensor, mC: cute.Tensor):
    total_elems = mA.numel()

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_activation_kernel(mA, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().cuda()
        C = torch.empty_like(x)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_activation_host, mA, mC)
            self.compiled[key] = compiled

        compiled(mA, mC)
        return C