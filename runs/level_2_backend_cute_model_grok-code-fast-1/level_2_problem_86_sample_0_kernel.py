import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_gelu_kernel(gA: cute.Tensor, gB: cute.Tensor, gBias: cute.Tensor, gDiv: cute.Tensor, gC: cute.Tensor): 
    tidx = cute.arch.thread_idx().x  
    bidx = cute.arch.block_idx().x  
    bdim = cute.arch.block_dim().x  

    thread_idx = bidx * bdim + tidx

    M, K = gA.shape
    N = gB.shape[1]
    total_elems = M * N

    if thread_idx >= total_elems:
        return

    i = thread_idx // N  
    j = thread_idx % N  

    sum_val = 0.0
    for k in range(K):
        sum_val += gA[i, k] * gB[k, j]

    sum_val += gBias[j]

    divisor = gDiv[0]
    x = sum_val / divisor

    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = 0.7978845608028654
    tanh_arg = sqrt_2_pi * (x + 0.044715 * x * x * x)
    gelu_val = 0.5 * x * (1.0 + cute.math.tanh(tanh_arg))

    gC[i, j] = gelu_val

@cute.jit
def fused_gemm_gelu_host(mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor, mDiv: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[1]
    total_elems = M * N
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_gemm_gelu_kernel(mA, mB, mBias, mDiv, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.compiled = {}

    def forward(self, x):
        x = x.contiguous().cuda()
        weight_t = self.linear.weight.t().contiguous().cuda()
        bias = self.linear.bias.contiguous().cuda()
        divisor_tensor = torch.tensor(self.divisor, dtype=torch.float32, device=x.device)

        M, K = x.shape
        N = weight_t.shape[1]
        C = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight_t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mDiv = from_dlpack(divisor_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_gelu_host, mA, mB, mBias, mDiv, mC)
            self.compiled[key] = compiled

        compiled(mA, mB, mBias, mDiv, mC)
        return C