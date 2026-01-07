import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_gemm_subtract_gavgpool_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gS: cute.Tensor,
    gOut: cute.Tensor, M: int, N: int, K: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy

    if row < M and col < N:
        acc = 0.0
        for k in range(K):
            acc += gX[row, k] * gW[col, k]
        if gB.shape[0] > 0:
            acc += gB[col]
        acc -= gS[col]
        gOut[row, col] = acc

@cute.kernel
def global_avg_pool_kernel(gIn: cute.Tensor, gOut: cute.Tensor, M: int, N: int):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    row = bidx * bdim + tidx
    if row < M:
        sum_val = 0.0
        for col in range(N):
            sum_val += gIn[row, col]
        gOut[row, 0] = sum_val / float(N)

@cute.kernel
def logsumexp_gelu_kernel(gIn: cute.Tensor, gOut: cute.Tensor, M: int):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    row = bidx * bdim + tidx
    if row < M:
        x = gIn[row, 0]
        max_val = x
        lse = math.log(math.exp(x - max_val)) + max_val
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = x * x * x
        tanh_arg = 0.7978845608 * (x + 0.044715 * x_cubed)
        # tanh approximation
        tanh_val = tanh_arg / (1.0 + abs(tanh_arg)) if abs(tanh_arg) < 4.0 else (1.0 if tanh_arg > 0 else -1.0)
        gelu_val = 0.5 * lse * (1.0 + tanh_val)
        gOut[row, 0] = gelu_val

@cute.kernel
def residual_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, M: int, N: int):
    tidx, tidy = cute.arch.thread_idx().x, cute.arch.thread_idx().y
    bidx, bidy = cute.arch.block_idx().x, cute.arch.block_idx().y
    bdimx, bdimy = cute.arch.block_dim().x, cute.arch.block_dim().y

    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy

    if row < M and col < N:
        gC[row, col] = gA[row, 0] + gB[row, col]

@cute.jit
def fused_gemm_subtract_gavgpool_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mS: cute.Tensor, mTmp: cute.Tensor,
    M: int, N: int, K: int
):
    threads_x = 16
    threads_y = 16
    grid_x = cute.ceil_div(M, threads_x)
    grid_y = cute.ceil_div(N, threads_y)
    fused_gemm_subtract_gavgpool_kernel(mX, mW, mB, mS, mTmp, M, N, K).launch(
        grid=(grid_x, grid_y, 1), block=(threads_x, threads_y, 1)
    )

@cute.jit
def global_avg_pool_host(mTmp: cute.Tensor, mPoolOut: cute.Tensor, M: int, N: int):
    threads = 256
    grid = cute.ceil_div(M, threads)
    global_avg_pool_kernel(mTmp, mPoolOut, M, N).launch(grid=(grid, 1, 1), block=(threads, 1, 1))

@cute.jit
def logsumexp_gelu_host(mPoolOut: cute.Tensor, mActOut: cute.Tensor, M: int):
    threads = 256
    grid = cute.ceil_div(M, threads)
    logsumexp_gelu_kernel(mPoolOut, mActOut, M).launch(grid=(grid, 1, 1), block=(threads, 1, 1))

@cute.jit
def residual_add_host(mActOut: cute.Tensor, mOrig: cute.Tensor, mFinal: cute.Tensor, M: int, N: int):
    threads_x = 16
    threads_y = 16
    grid_x = cute.ceil_div(M, threads_x)
    grid_y = cute.ceil_div(N, threads_y)
    residual_add_kernel(mActOut, mOrig, mFinal, M, N).launch(
        grid=(grid_x, grid_y, 1), block=(threads_x, threads_y, 1)
    )

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x.clone().detach()
        M, K = x.shape
        N = self.gemm.out_features

        x_contig = x.contiguous().cuda()
        W = self.gemm.weight.data.contiguous().cuda()
        B = self.gemm.bias.data.contiguous().cuda() if self.gemm.bias is not None else torch.empty(0, device=x.device)
        S = self.subtract.data.contiguous().cuda()

        tmp = torch.empty((M, N), dtype=x.dtype, device=x.device)
        pool_out = torch.empty((M, 1), dtype=x.dtype, device=x.device)
        act_out = torch.empty((M, 1), dtype=x.dtype, device=x.device)
        final_out = torch.empty_like(x_contig)

        mX = from_dlpack(x_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mS = from_dlpack(S, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mTmp = from_dlpack(tmp, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mPoolOut = from_dlpack(pool_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mActOut = from_dlpack(act_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mOrig = from_dlpack(original_x.contiguous().cuda(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mFinal = from_dlpack(final_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = {
                'fused_gemm_subtract_gavgpool': cute.compile(fused_gemm_subtract_gavgpool_host, mX, mW, mB, mS, mTmp, M, N, K),
                'global_avg_pool': cute.compile(global_avg_pool_host, mTmp, mPoolOut, M, N),
                'logsumexp_gelu': cute.compile(logsumexp_gelu_host, mPoolOut, mActOut, M),
                'residual_add': cute.compile(residual_add_host, mActOut, mOrig, mFinal, M, K)
            }
            self.compiled[key] = compiled

        compiled['fused_gemm_subtract_gavgpool'](mX, mW, mB, mS, mTmp, M, N, K)
        compiled['global_avg_pool'](mTmp, mPoolOut, M, N)
        compiled['logsumexp_gelu'](mPoolOut, mActOut, M)
        compiled['residual_add'](mActOut, mOrig, mFinal, M, K)

        return final_out