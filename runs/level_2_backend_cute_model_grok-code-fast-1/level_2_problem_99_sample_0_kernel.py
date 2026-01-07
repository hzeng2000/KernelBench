import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_linear_gelu_kernel(gA: cute.Tensor, gB: cute.Tensor, gBias: cute.Tensor, gC: cute.Tensor): 
    tidx = cute.arch.thread_idx().x  
    bidx = cute.arch.block_idx().x  
    bdim = cute.arch.block_dim().x  

    thread_idx = bidx * bdim + tidx

    M, K = gA.shape
    _, N = gB.shape
    total_elems = M * N
    if thread_idx >= total_elems:
        return
    mi = thread_idx // N  
    ni = thread_idx % N  

    sum_val = 0.0
    for k in range(K):
        sum_val += gA[mi, k] * gB[k, ni]
    sum_val += gBias[ni]

    # GELU approximation
    gelu_val = 0.5 * sum_val * (1 + math.tanh(math.sqrt(2 / math.pi) * (sum_val + 0.044715 * sum_val * sum_val * sum_val)))
    gC[mi, ni] = gelu_val

@cute.jit
def fused_linear_gelu_host(mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_linear_gelu_kernel(mA, mB, mBias, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M, K = x.shape
        N = self.linear.out_features
        x = x.contiguous().cuda()
        W_t = self.linear.weight.t().contiguous().cuda()
        b = self.linear.bias.contiguous().cuda()
        C = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(W_t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(b, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_linear_gelu_host, mA, mB, mBias, mC)
            self.compiled[key] = compiled

        compiled(mA, mB, mBias, mC)
        # Apply softmax
        C = torch.nn.functional.softmax(C, dim=1)
        return C