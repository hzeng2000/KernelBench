import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_relu_div_kernel(gA: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, divisor: float):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    thread_idx = bidx * bdim + tidx

    M, K = gA.shape
    N = gW.shape[0]
    total = M * N

    if thread_idx < total:
        mi = thread_idx // N
        ni = thread_idx % N
        accum = 0.0
        for ki in range(K):
            accum += gA[mi, ki] * gW[ni, ki]
        accum += gB[ni]
        accum = max(accum, 0.0)
        gC[mi, ni] = accum / divisor

@cute.jit
def fused_gemm_relu_div_host(mA: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, divisor: float):
    M = mA.shape[0]
    N = mW.shape[0]
    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    fused_gemm_relu_div_kernel(mA, mW, mB, mC, divisor).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor
        self.W = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.b = torch.nn.Parameter(torch.randn(out_features))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M, K = x.shape
        N = self.out_features
        A = x.contiguous().cuda()
        W = self.W.contiguous().cuda()
        B = self.b.contiguous().cuda()
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (A.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_relu_div_host, mA, mW, mB, mC, self.divisor)
            self.compiled[key] = compiled

        compiled(mA, mW, mB, mC, self.divisor)
        return C