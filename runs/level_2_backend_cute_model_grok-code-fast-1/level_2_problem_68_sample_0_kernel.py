import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_min_sub_kernel(gX: cute.Tensor, c: float, gY: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gX.shape
    total = m * n
    if thread_idx < total:
        ni = thread_idx % n  
        mi = thread_idx // n  

        val = gX[mi, ni]
        gY[mi, ni] = min(val, c) - c

@cute.jit
def fused_min_sub_host(mX: cute.Tensor, c: float, mY: cute.Tensor):
    M, N = mX.shape

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_min_sub_kernel(mX, c, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.compiled = {}

    def forward(self, x):
        x = self.linear(x)
        x = x.contiguous().cuda()
        c = self.constant.item()
        y = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_min_sub_host, mX, c, mY)
            self.compiled[key] = compiled

        compiled(mX, c, mY)
        return y