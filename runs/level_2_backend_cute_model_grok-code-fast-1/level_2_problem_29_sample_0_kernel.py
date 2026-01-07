import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_mish_kernel(gA: cute.Tensor, gB: cute.Tensor, gBias: cute.Tensor, gC: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    thread_idx = bidx * bdim + tidx

    M, K = gA.shape
    N, _ = gB.shape

    if thread_idx >= M * N:
        return

    mi = thread_idx // N
    ni = thread_idx % N

    sum_val = 0.0
    for ki in range(K):
        sum_val += gA[mi, ki] * gB[ni, ki]

    x = sum_val + gBias[ni]

    # mish
    softplus = cute.log(1 + cute.exp(x))
    tanh_sp = cute.tanh(softplus)
    x = x * tanh_sp

    # mish again
    softplus = cute.log(1 + cute.exp(x))
    tanh_sp = cute.tanh(softplus)
    x = x * tanh_sp

    gC[mi, ni] = x

@cute.jit
def fused_gemm_mish_host(mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    threads_per_block = 256
    total = M * N
    grid_x = cute.ceil_div(total, threads_per_block)
    fused_gemm_mish_kernel(mA, mB, mBias, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        M, K = x.shape
        N = weight.shape[0]
        C = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_mish_host, mA, mB, mBias, mC)
            self.compiled[key] = compiled

        compiled(mA, mB, mBias, mC)
        return C