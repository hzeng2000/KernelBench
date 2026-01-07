import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_subtract_mean_gelu_kernel(gX: cute.Tensor, gSubtract: cute.Tensor, gC: cute.Tensor):
    i = cute.arch.block_idx(0)
    tid = cute.arch.thread_idx(0)
    bdim = cute.arch.block_dim(0)
    batch, out_features = gX.shape
    if i >= batch:
        return
    shared_sum = cute.shared_memory(float, (bdim,))
    local_sum = 0.0
    for j in range(tid, out_features, bdim):
        val = gX[i, j] - gSubtract[j]
        local_sum += val
    shared_sum[tid] = local_sum
    cute.sync()
    for s in range(bdim // 2, 0, s // 2):
        if tid < s:
            shared_sum[tid] += shared_sum[tid + s]
        cute.sync()
    if tid == 0:
        mean = shared_sum[0] / out_features
        gC[i, 0] = 0.5 * mean * (1 + cute.math.erf(mean / cute.math.sqrt(2.0)))

@cute.jit
def fused_subtract_mean_gelu_host(mX: cute.Tensor, mSubtract: cute.Tensor, mC: cute.Tensor):
    batch = mX.shape[0]
    threads_per_block = 256
    grid_x = batch
    fused_subtract_mean_gelu_kernel(mX, mSubtract, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.compiled = {}

    def forward(self, x):
        x = self.gemm(x)
        x = x.contiguous().cuda()
        subtract = self.subtract.contiguous().cuda()
        batch_size = x.shape[0]
        c = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mSubtract = from_dlpack(subtract, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(c, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_subtract_mean_gelu_host, mX, mSubtract, mC)
            self.compiled[key] = compiled
        compiled(mX, mSubtract, mC)
        return c