import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_activation_kernel(gX: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    b, c, h, w = gX.shape
    total = b * c * h * w

    if thread_idx >= total:
        return

    stride_b = c * h * w
    stride_c = h * w
    stride_h = w

    bi = thread_idx // stride_b
    rem = thread_idx % stride_b
    ci = rem // stride_c
    rem = rem % stride_c
    hi = rem // stride_h
    wi = rem % stride_h

    x_val = gX[bi, ci, hi, wi]

    # HardSwish: x * relu6(x + 3) / 6
    relu6_val = min(max(x_val + 3.0, 0.0), 6.0)
    hs_val = x_val * relu6_val / 6.0

    # Then ReLU (though redundant, as HardSwish >= 0)
    result = max(hs_val, 0.0)

    gC[bi, ci, hi, wi] = result

@cute.jit
def fused_activation_host(mX: cute.Tensor, mC: cute.Tensor):
    b, c, h, w = mX.shape
    total_elems = b * c * h * w

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_activation_kernel(mX, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.contiguous().cuda()
        C = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_activation_host, mX, mC)
            self.compiled[key] = compiled

        compiled(mX, mC)
        return C