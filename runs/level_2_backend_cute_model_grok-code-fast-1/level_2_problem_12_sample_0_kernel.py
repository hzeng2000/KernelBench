import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_mul_leakyrelu_kernel(gA: cute.Tensor, gC: cute.Tensor, multiplier: float, negative_slope: float): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gA.shape
    total_elems = m * n
    if thread_idx >= total_elems:
        return

    ni = thread_idx % n  
    mi = thread_idx // n  

    a_val = gA[mi, ni]
    c_val = a_val * multiplier
    gC[mi, ni] = c_val if c_val >= 0.0 else c_val * negative_slope

@cute.jit
def fused_mul_leakyrelu_host(mA: cute.Tensor, mC: cute.Tensor, multiplier: float, negative_slope: float):
    M = mA.shape[0]
    N = mA.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_mul_leakyrelu_kernel(mA, mC, multiplier, negative_slope).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model that performs a Gemm, and applies fused multiplication and LeakyReLU using a custom CuTe kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        self.compiled = {}

    def forward(self, x):
        x = self.gemm(x)
        M, N = x.shape
        x = x.contiguous().cuda()
        C = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_mul_leakyrelu_host, mA, mC, self.multiplier, self.negative_slope)
            self.compiled[key] = compiled

        compiled(mA, mC, self.multiplier, self.negative_slope)
        return C