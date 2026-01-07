import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_kernel(gX: cute.Tensor, gS: cute.Tensor, constant: float, gC: cute.Tensor):
    bidx = cute.arch.block_idx().x
    tidx = cute.arch.thread_idx().x
    bdim = cute.arch.block_dim().x
    
    shared = cute.shared_memory(float, (bdim,))
    
    local_sum = 0.0
    for k in cute.range(tidx, gX.shape[1], bdim):
        local_sum += gX[bidx, k] * gS[k]
    
    shared[tidx] = local_sum
    cute.arch.barrier()
    
    step = bdim // 2
    while step > 0:
        if tidx < step:
            shared[tidx] += shared[tidx + step]
        cute.arch.barrier()
        step //= 2
    
    if tidx == 0:
        gC[bidx, 0] = shared[0] * constant

@cute.jit
def fused_host(mX: cute.Tensor, mS: cute.Tensor, constant: float, mC: cute.Tensor):
    batch_size = mX.shape[0]
    threads_per_block = 256
    grid_x = batch_size
    
    fused_kernel(mX, mS, constant, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x):
        s = torch.sum(self.weight, dim=0)
        constant = self.scaling_factor * 0.5
        x = x.contiguous().cuda()
        s = s.contiguous().cuda()
        c = torch.empty((x.shape[0], 1), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mS = from_dlpack(s, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(c, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_host, mX, mS, constant, mC)
            self.compiled[key] = compiled
        
        compiled(mX, mS, constant, mC)
        return c