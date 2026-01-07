import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def logsumexp_relu_kernel(gX: cute.Tensor, gY: cute.Tensor, c: int):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    
    b_size = gX.shape[0]
    d_size = gX.shape[2]
    h_size = gX.shape[3]
    w_size = gX.shape[4]
    
    total_spatial = d_size * h_size * w_size
    total_outputs = b_size * total_spatial
    
    bi = bidx // total_spatial
    spatial_idx = bidx % total_spatial
    di = spatial_idx // (h_size * w_size)
    hw_idx = spatial_idx % (h_size * w_size)
    hi = hw_idx // w_size
    wi = hw_idx % w_size
    
    val = gX[bi, tidx, di, hi, wi] if tidx < c else -3.4028235e+38
    
    s_data = cute.shared_tensor(cute.float32, (c,))
    s_data[tidx] = val
    cute.sync()
    
    max_val = -3.4028235e+38
    for i in cute.range(c):
        max_val = cute.max(max_val, s_data[i])
    
    sum_exp = 0.0
    for i in cute.range(c):
        sum_exp += cute.exp(s_data[i] - max_val)
    
    result = cute.log(sum_exp) + max_val
    result = cute.max(result, 0.0)
    
    if tidx == 0:
        gY[bi, 0, di, hi, wi] = result

@cute.jit
def logsumexp_relu_host(mX: cute.Tensor, mY: cute.Tensor, c: int):
    b = mX.shape[0]
    d = mX.shape[2]
    h = mX.shape[3]
    w = mX.shape[4]
    total_outputs = b * d * h * w
    threads_per_block = c
    grid_x = total_outputs
    logsumexp_relu_kernel(mX, mY, c).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.compiled = {}
        self.c = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        b, c, d, h, w = x.shape
        x = x.contiguous().cuda()
        y = torch.empty((b, 1, d, h, w), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key = (x.dtype, c)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(logsumexp_relu_host, mX, mY, self.c)
            self.compiled[key] = compiled
        compiled(mX, mY, self.c)
        return y