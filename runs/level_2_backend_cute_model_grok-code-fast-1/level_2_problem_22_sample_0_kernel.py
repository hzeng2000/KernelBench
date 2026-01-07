import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def scale_residual_clamp_kernel(gIn: cute.Tensor, gOut: cute.Tensor, scale: float, min_val: float, max_val: float): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gIn.shape
    total = m * n
    if thread_idx >= total:
        return
    ni = thread_idx % n  
    mi = thread_idx // n  

    val = gIn[mi, ni]
    val = val * scale * 2.0
    val = cute.max(min_val, cute.min(max_val, val))

    gOut[mi, ni] = val

@cute.jit
def scale_residual_clamp_host(mIn: cute.Tensor, mOut: cute.Tensor, scale: float, min_val: float, max_val: float):
    M = mIn.shape[0]
    N = mIn.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    scale_residual_clamp_kernel(mIn, mOut, scale, min_val, max_val).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def logsumexp_kernel(gIn: cute.Tensor, gOut: cute.Tensor):
    batch, hidden = gIn.shape
    tid = cute.arch.thread_idx().x
    bid = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    row = bid
    if row >= batch:
        return

    shared_max = cute.shared_memory(float, (bdim,))
    shared_sum = cute.shared_memory(float, (bdim,))

    num_per_thread = cute.ceil_div(hidden, bdim)
    local_max = float('-inf')
    for i in range(num_per_thread):
        idx = tid * num_per_thread + i
        if idx < hidden:
            val = gIn[row, idx]
            local_max = cute.max(local_max, val)
    shared_max[tid] = local_max
    cute.sync()
    for s in range(1, bdim):
        if tid % (2 * s) == 0 and tid + s < bdim:
            shared_max[tid] = cute.max(shared_max[tid], shared_max[tid + s])
        cute.sync()
    global_max = shared_max[0]

    local_sum = 0.0
    for i in range(num_per_thread):
        idx = tid * num_per_thread + i
        if idx < hidden:
            val = gIn[row, idx]
            local_sum += cute.exp(val - global_max)
    shared_sum[tid] = local_sum
    cute.sync()
    for s in range(1, bdim):
        if tid % (2 * s) == 0 and tid + s < bdim:
            shared_sum[tid] += shared_sum[tid + s]
        cute.sync()
    if tid == 0:
        gOut[row, 0] = cute.log(shared_sum[0]) + global_max

@cute.jit
def logsumexp_host(mIn: cute.Tensor, mOut: cute.Tensor):
    batch = mIn.shape[0]
    block_size = 256
    grid_x = batch
    logsumexp_kernel(mIn, mOut).launch(grid=(grid_x, 1, 1), block=(block_size, 1, 1))

@cute.kernel
def mish_multiply_kernel(gIn: cute.Tensor, gOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gIn.shape
    total = m * n
    if thread_idx >= total:
        return
    ni = thread_idx % n  
    mi = thread_idx // n  

    val = gIn[mi, ni]
    softplus = cute.log(1.0 + cute.exp(val))
    mish_val = val * cute.tanh(softplus)
    gOut[mi, ni] = val * mish_val

@cute.jit
def mish_multiply_host(mIn: cute.Tensor, mOut: cute.Tensor):
    M = mIn.shape[0]
    N = mIn.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    mish_multiply_kernel(mIn, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(hidden_size, input_size))
        self.b = torch.nn.Parameter(torch.randn(hidden_size))
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda()
        batch, in_size = x.shape
        out_size = self.W.shape[0]

        # GEMM with bias
        C = torch.empty((batch, out_size), dtype=x.dtype, device=x.device)
        C.copy_(self.b.unsqueeze(0).expand(batch, -1))
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.W.t(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        cute.gemm(mX, mW, mC, accum=cute.gemm.accum(alpha=1.0, beta=1.0))

        # Scale, residual, clamp
        clamped = torch.empty_like(C)
        mClamped = from_dlpack(clamped, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(scale_residual_clamp_host, mC, mClamped, self.scale_factor, self.clamp_min, self.clamp_max)
            self.compiled[key] = compiled
        compiled(mC, mClamped, self.scale_factor, self.clamp_min, self.clamp_max)

        # LogSumExp
        lse = torch.empty((batch, 1), dtype=x.dtype, device=x.device)
        mLse = from_dlpack(lse, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        compiled_lse = self.compiled.get('lse')
        if compiled_lse is None:
            compiled_lse = cute.compile(logsumexp_host, mClamped, mLse)
            self.compiled['lse'] = compiled_lse
        compiled_lse(mClamped, mLse)

        # Mish multiply
        final = torch.empty_like(lse)
        mFinal = from_dlpack(final, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        compiled_mish = self.compiled.get('mish')
        if compiled_mish is None:
            compiled_mish = cute.compile(mish_multiply_host, mLse, mFinal)
            self.compiled['mish'] = compiled_mish
        compiled_mish(mLse, mFinal)
        return final