import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gelu_avg_kernel(gX: cute.Tensor, gY: cute.Tensor, H: int, W: int):
    bidx = cute.arch.block_idx(0)
    tid = cute.arch.thread_idx(0)
    
    batch_size, out_channels = gY.shape
    batch_idx = bidx // out_channels
    c_idx = bidx % out_channels
    
    s_sum = cute.shared_memory(float, (256,))
    s_sum[tid] = 0.0
    
    num_spatial = H * W
    elems_per_thread = cute.ceil_div(num_spatial, 256)
    
    for i in range(elems_per_thread):
        idx = tid * elems_per_thread + i
        if idx < num_spatial:
            h = idx // W
            w = idx % W
            val = gX[batch_idx, c_idx, h, w]
            gelu_val = 0.5 * val * (1.0 + cute.math.erf(val * 0.7071067811865476))  # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
            s_sum[tid] += gelu_val
    
    cute.sync()
    
    # Tree reduction
    for d in [128, 64, 32, 16, 8, 4, 2, 1]:
        if tid < d:
            s_sum[tid] += s_sum[tid + d]
        cute.sync()
    
    if tid == 0:
        gY[batch_idx, c_idx] = s_sum[0] / num_spatial

@cute.jit
def gelu_avg_host(mX: cute.Tensor, mY: cute.Tensor, H: int, W: int):
    batch_size, out_channels = mY.shape
    grid_x = batch_size * out_channels
    threads_per_block = 256
    
    gelu_avg_kernel(mX, mY, H, W).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        batch_size, out_channels, H, W = x.shape
        x = x.contiguous().cuda()
        y = torch.empty((batch_size, out_channels), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key = (x.dtype, H, W)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(gelu_avg_host, mX, mY, H, W)
            self.compiled[key] = compiled
        
        compiled(mX, mY, H, W)
        return y