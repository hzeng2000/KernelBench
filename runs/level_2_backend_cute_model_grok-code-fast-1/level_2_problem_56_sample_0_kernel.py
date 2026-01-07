import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, gBias: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx
    M, K = gA.shape
    _, N = gB.shape
    total_threads = M * N
    if thread_idx < total_threads:
        mi = thread_idx // N
        ni = thread_idx % N
        sum_val = 0.0
        for k in range(K):
            sum_val += gA[mi, k] * gB[k, ni]
        gC[mi, ni] = sum_val + gBias[ni]

@cute.jit
def gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[1]
    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    gemm_kernel(mA, mB, mC, mBias).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def sigmoid_kernel(gA: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    m, n = gA.shape
    total = m * n
    if thread_idx < total:
        mi = thread_idx // n
        ni = thread_idx % n
        gC[mi, ni] = 1.0 / (1.0 + cute.exp(-gA[mi, ni]))

@cute.jit
def sigmoid_host(mA: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mA.shape[1]
    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    sigmoid_kernel(mA, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def sum_kernel(gA: cute.Tensor, gC: cute.Tensor):
    bidx = cute.arch.block_idx(0)
    tidx = cute.arch.thread_idx(0)
    bdim = cute.arch.block_dim(0)
    M, N = gA.shape
    if bidx < M:
        sum_val = 0.0
        for i in range(tidx, N, bdim):
            sum_val += gA[bidx, i]
        shared = cute.shared_memory(float, bdim)
        shared[tidx] = sum_val
        cute.sync()
        s = bdim // 2
        while s > 0:
            if tidx < s:
                shared[tidx] += shared[tidx + s]
            cute.sync()
            s //= 2
        if tidx == 0:
            gC[bidx, 0] = shared[0]

@cute.jit
def sum_host(mA: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    threads_per_block = 256
    sum_kernel(mA, mC).launch(grid=(M, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight = torch.randn(hidden_size, input_size, dtype=torch.float32).cuda()
        self.bias = torch.randn(hidden_size, dtype=torch.float32).cuda()
        self.compiled_gemm = {}
        self.compiled_sigmoid = {}
        self.compiled_sum = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_size = x.shape
        hidden_size = self.weight.shape[0]
        x = x.contiguous().cuda()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous()
        temp = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=x.device)
        output = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=x.device)
        final = torch.empty((batch_size, 1), dtype=torch.float32, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mTemp = from_dlpack(temp, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mFinal = from_dlpack(final, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled_gemm = self.compiled_gemm.get(key)
        if compiled_gemm is None:
            compiled_gemm = cute.compile(gemm_host, mX, mW, mTemp, mBias)
            self.compiled_gemm[key] = compiled_gemm
        compiled_gemm(mX, mW, mTemp, mBias)

        compiled_sigmoid = self.compiled_sigmoid.get(key)
        if compiled_sigmoid is None:
            compiled_sigmoid = cute.compile(sigmoid_host, mTemp, mOutput)
            self.compiled_sigmoid[key] = compiled_sigmoid
        compiled_sigmoid(mTemp, mOutput)

        compiled_sum = self.compiled_sum.get(key)
        if compiled_sum is None:
            compiled_sum = cute.compile(sum_host, mOutput, mFinal)
            self.compiled_sum[key] = compiled_sum
        compiled_sum(mOutput, mFinal)

        return final