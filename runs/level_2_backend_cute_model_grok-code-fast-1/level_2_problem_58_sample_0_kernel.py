import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def post_process_kernel(gX: cute.Tensor, gBias: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    
    B, C, D, H, W = gX.shape
    spatial_total = D * H * W
    total_spatial = B * spatial_total
    
    spatial_idx = bidx
    if spatial_idx >= total_spatial:
        return
    
    b = spatial_idx // spatial_total
    rem = spatial_idx % spatial_total
    d = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W
    
    c = tidx
    if c >= C:
        return
    
    x_val = gX[b, c, d, h, w]
    
    shared = cute.shared_memory(float, (C,))
    shared[c] = x_val
    cute.syncthreads()
    
    if c == 0:
        max_val = shared[0]
        for i in range(1, C):
            max_val = cute.max(max_val, shared[i])
        
        sum_val = 0.0
        for i in range(C):
            sum_val += cute.exp(shared[i] - max_val)
        
        lse = cute.log(sum_val) + max_val
        
        sig = 1.0 / (1.0 + cute.exp(-(lse + 3.0)))
        hs = lse * sig / 6.0
        
        bias_val = gBias[0, 0, 0, 0]
        hs -= bias_val
        
        hs = cute.max(-1.0, cute.min(1.0, hs))
        
        gY[b, 0, d, h, w] = hs

@cute.jit
def post_process_host(mX: cute.Tensor, mBias: cute.Tensor, mY: cute.Tensor):
    B, C, D, H, W = mX.shape
    spatial_total = B * D * H * W
    threads_per_block = C
    grid_x = spatial_total
    
    post_process_kernel(mX, mBias, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, fused LogSumExp, HardSwish, subtraction, clamp operations in a custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        y = torch.empty((B, 1, D, H, W), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(post_process_host, mX, mBias, mY)
            self.compiled[key] = compiled
        
        compiled(mX, mBias, mY)
        return y