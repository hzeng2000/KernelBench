import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor): 
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx
    M, K = gA.shape
    _, N = gB.shape
    total = M * N
    if thread_idx < total:
        mi = thread_idx // N
        ni = thread_idx % N
        sum_val = 0.0
        for k in range(K):
            sum_val += gA[mi, k] * gB[k, ni]
        gC[mi, ni] = sum_val

@cute.jit
def gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[1]
    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    gemm_kernel(mA, mB, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def add_bias_kernel(gC: cute.Tensor, gBias: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx
    M, N = gC.shape
    total = M * N
    if thread_idx < total:
        mi = thread_idx // N
        ni = thread_idx % N
        gC[mi, ni] += gBias[ni]

@cute.jit
def add_bias_host(mC: cute.Tensor, mBias: cute.Tensor):
    M = mC.shape[0]
    N = mC.shape[1]
    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    add_bias_kernel(mC, mBias).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def layernorm_kernel(gX: cute.Tensor, gC: cute.Tensor, eps: float):
    bidx = cute.arch.block_idx(0)
    b = bidx
    M, N = gX.shape
    sum_val = 0.0
    for f in range(N):
        sum_val += gX[b, f]
    mean = sum_val / N
    sum_var = 0.0
    for f in range(N):
        diff = gX[b, f] - mean
        sum_var += diff * diff
    var = sum_var / N
    std = cute.sqrt(var + eps)
    for f in range(N):
        gC[b, f] = (gX[b, f] - mean) / std

@cute.jit
def layernorm_host(mX: cute.Tensor, mC: cute.Tensor, eps: float):
    M = mX.shape[0]
    layernorm_kernel(mX, mC, eps).launch(grid=(M, 1, 1), block=(1, 1, 1))

@cute.kernel
def elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor): 
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx
    m, n = gA.shape
    total = m * n
    if thread_idx < total:
        mi = thread_idx // n
        ni = thread_idx % n
        gC[mi, ni] = gA[mi, ni] + gB[mi, ni]

@cute.jit
def elementwise_add_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mA.shape[1]
    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    elementwise_add_kernel(mA, mB, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def elementwise_mul_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor): 
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx
    m, n = gA.shape
    total = m * n
    if thread_idx < total:
        mi = thread_idx // n
        ni = thread_idx % n
        gC[mi, ni] = gA[mi, ni] * gB[mi, ni]

@cute.jit
def elementwise_mul_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mA.shape[1]
    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    elementwise_mul_kernel(mA, mB, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.compiled = {}

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        y = y.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        C = torch.empty((batch_size, self.out_features), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(weight.T.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key_gemm = (x.dtype,)
        compiled_gemm = self.compiled.get('gemm')
        if compiled_gemm is None:
            compiled_gemm = cute.compile(gemm_host, mX, mW, mC)
            self.compiled['gemm'] = compiled_gemm
        compiled_gemm(mX, mW, mC)
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        key_bias = (C.dtype,)
        compiled_bias = self.compiled.get('bias')
        if compiled_bias is None:
            compiled_bias = cute.compile(add_bias_host, mC, mBias)
            self.compiled['bias'] = compiled_bias
        compiled_bias(mC, mBias)
        C_norm = torch.empty_like(C)
        mC_in = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC_out = from_dlpack(C_norm, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key_norm = (C.dtype, self.eps)
        compiled_norm = self.compiled.get('norm')
        if compiled_norm is None:
            compiled_norm = cute.compile(layernorm_host, mC_in, mC_out, self.eps)
            self.compiled['norm'] = compiled_norm
        compiled_norm(mC_in, mC_out, self.eps)
        C_add = torch.empty_like(C_norm)
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC_add = from_dlpack(C_add, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key_add = (C_norm.dtype,)
        compiled_add = self.compiled.get('add')
        if compiled_add is None:
            compiled_add = cute.compile(elementwise_add_host, mC_out, mY, mC_add)
            self.compiled['add'] = compiled_add
        compiled_add(mC_out, mY, mC_add)
        C_mul = torch.empty_like(C_add)
        mC_mul = from_dlpack(C_mul, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key_mul = (C_add.dtype,)
        compiled_mul = self.compiled.get('mul')
        if compiled_mul is None:
            compiled_mul = cute.compile(elementwise_mul_host, mC_add, mY, mC_mul)
            self.compiled['mul'] = compiled_mul
        compiled_mul(mC_add, mY, mC_mul)
        return C_mul