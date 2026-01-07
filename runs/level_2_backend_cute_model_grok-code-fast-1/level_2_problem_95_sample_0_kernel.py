import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_activation_kernel(gX: cute.Tensor, gAdd: cute.Tensor, gOut: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    M, N = gX.shape
    total_elems = M * N
    if thread_idx >= total_elems:
        return

    ni = thread_idx % N  
    mi = thread_idx // N  

    val = gX[mi, ni] + gAdd[ni]

    # Swish
    sigmoid_val = 1.0 / (1.0 + math.exp(-val))
    val = sigmoid_val * val

    # Tanh
    val = math.tanh(val)

    # GELU (approximation)
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    val = 0.5 * val * (1.0 + math.tanh(sqrt_2_pi * (val + 0.044715 * val * val * val)))

    # Hardtanh
    val = max(-1.0, min(1.0, val))

    gOut[mi, ni] = val

@cute.jit
def fused_activation_host(mX: cute.Tensor, mAdd: cute.Tensor, mOut: cute.Tensor):
    M = mX.shape[0]
    N = mX.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_activation_kernel(mX, mAdd, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features, bias=False)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 
        self.compiled = {}

    def forward(self, x):
        x = self.matmul(x)
        M, N = x.shape
        x = x.contiguous().cuda()
        add_value = self.add_value.contiguous().cuda()
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mAdd = from_dlpack(add_value, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_activation_host, mX, mAdd, mOut)
            self.compiled[key] = compiled

        compiled(mX, mAdd, mOut)
        return out