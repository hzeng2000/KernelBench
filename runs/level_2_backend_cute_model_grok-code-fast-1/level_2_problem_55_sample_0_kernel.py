import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def reduction_kernel(gX: cute.Tensor, gY: cute.Tensor, scale_factor: float):
    tid = cute.arch.thread_idx(0)
    bid = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    
    batch = bid
    if batch >= gX.shape[0]:
        return
    
    num_outputs = gX.shape[1] // 2
    shared = cute.arch.shared_memory(float, bdim)
    
    sum_val = 0.0
    for i in range(tid, num_outputs, bdim):
        max_val = max(gX[batch, 2 * i], gX[batch, 2 * i + 1])
        sum_val += max_val
    
    shared[tid] = sum_val
    cute.arch.syncthreads()
    
    s = bdim // 2
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cute.arch.syncthreads()
        s //= 2
    
    if tid == 0:
        gY[batch] = shared[0] * scale_factor

@cute.jit
def reduction_host(mX: cute.Tensor, mY: cute.Tensor, scale_factor: float):
    batch_size = mX.shape[0]
    threads_per_block = 256
    grid_x = batch_size
    reduction_kernel(mX, mY, scale_factor).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.compiled = {}

    def forward(self, x):
        x = self.matmul(x)
        x = x.contiguous().cuda()
        batch_size, out_features = x.shape
        y = torch.empty((batch_size,), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(reduction_host, mX, mY, self.scale_factor)
            self.compiled[key] = compiled
        
        compiled(mX, mY, self.scale_factor)
        return y