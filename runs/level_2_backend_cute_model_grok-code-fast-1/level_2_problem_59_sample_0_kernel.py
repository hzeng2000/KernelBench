import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def swish_scale_kernel(gA: cute.Tensor, gScaling: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gA.shape
    total = m * n
    if thread_idx >= total:
        return

    ni = thread_idx % n  
    mi = thread_idx // n  

    a_val = gA[mi, ni]
    scaling_val = gScaling[0]
    sig_val = 1 / (1 + cute.exp(-a_val))
    gC[mi, ni] = a_val * sig_val * scaling_val

@cute.jit
def swish_scale_host(mA: cute.Tensor, mScaling: cute.Tensor, mC: cute.Tensor):
    M, N = mA.shape

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    swish_scale_kernel(mA, mScaling, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x):
        x = self.matmul(x)
        M, N = x.shape
        x = x.contiguous().cuda()
        scaling = torch.tensor(self.scaling_factor, dtype=x.dtype, device=x.device)
        C = torch.empty_like(x)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mScaling = from_dlpack(scaling, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(swish_scale_host, mA, mScaling, mC)
            self.compiled[key] = compiled

        compiled(mA, mScaling, mC)
        return C