import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_gemm_kernel(mA: cute.Tensor, mW: cute.Tensor, mBias: cute.Tensor, mC: cute.Tensor, scaling_factor: float, hardtanh_min: float, hardtanh_max: float):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    thread_id = bidx * bdim + tidx

    batch, out_features = mC.shape
    in_features = mA.shape[1]
    total = batch * out_features

    if thread_id >= total:
        return

    b = thread_id // out_features
    o = thread_id % out_features

    sum_val = 0.0
    for i in range(in_features):
        sum_val += mA[b, i] * mW[o, i]

    sum_val += mBias[o]
    sum_val *= scaling_factor
    sum_val = max(hardtanh_min, min(hardtanh_max, sum_val))
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = math.sqrt(2.0)
    erf_arg = sum_val / sqrt_2
    # Approximate erf using a polynomial or assume cute has it; for simplicity, use a simple approx
    # erf(x) ≈ tanh(1.27324 * x) for x in [-1,1], but for general, use built-in if available
    # Assuming cute.math.erf is available
    erf_val = cute.math.erf(erf_arg)
    sum_val = 0.5 * sum_val * (1.0 + erf_val)

    mC[b, o] = sum_val

@cute.jit
def fused_gemm_host(mA: cute.Tensor, mW: cute.Tensor, mBias: cute.Tensor, mC: cute.Tensor, scaling_factor: float, hardtanh_min: float, hardtanh_max: float):
    batch, out_features = mC.shape
    total_elems = batch * out_features
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_gemm_kernel(mA, mW, mBias, mC, scaling_factor, hardtanh_min, hardtanh_max).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.compiled = {}

    def forward(self, x):
        batch_size, in_features = x.shape
        out_features = self.gemm.out_features
        x = x.contiguous().cuda()
        W = self.gemm.weight.contiguous().cuda()
        bias = self.gemm.bias.contiguous().cuda()
        C = torch.empty((batch_size, out_features), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_host, mA, mW, mBias, mC, self.scaling_factor, self.hardtanh_min, self.hardtanh_max)
            self.compiled[key] = compiled

        compiled(mA, mW, mBias, mC, self.scaling_factor, self.hardtanh_min, self.hardtanh_max)
        return C