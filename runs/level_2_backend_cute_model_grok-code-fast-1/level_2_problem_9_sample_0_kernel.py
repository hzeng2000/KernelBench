import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_elementwise_kernel(gA: cute.Tensor, gC: cute.Tensor, subtract_value: float, multiply_value: float): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gA.shape
    ni = thread_idx % n  
    mi = thread_idx // n  

    a_val = gA[mi, ni]
    c_val = (a_val - subtract_value) * multiply_value
    gC[mi, ni] = torch.relu(c_val)

@cute.jit
def fused_elementwise_host(mA: cute.Tensor, mC: cute.Tensor, subtract_value: float, multiply_value: float):
    M = mA.shape[0]
    N = mA.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_elementwise_kernel(mA, mC, subtract_value, multiply_value).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, and fused subtraction, multiplication, and ReLU activation in a single custom kernel.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.compiled = {}

    def forward(self, x):
        x = self.linear(x)
        M, N = x.shape
        x = x.contiguous().cuda()
        C = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_elementwise_host, mA, mC, self.subtract_value, self.multiply_value)
            self.compiled[key] = compiled

        compiled(mA, mC, self.subtract_value, self.multiply_value)
        return C