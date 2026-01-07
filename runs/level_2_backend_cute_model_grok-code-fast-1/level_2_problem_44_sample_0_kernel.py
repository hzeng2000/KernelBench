import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def multiply_and_pool_kernel(gA: cute.Tensor, multiplier: float, num: int, gC: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    
    batch = gA.shape[0]
    channels = gA.shape[1]
    H = gA.shape[2]
    W = gA.shape[3]
    
    batch_idx = bidx // channels
    channel_idx = bidx % channels
    
    total_elems = H * W
    threads_per_block = 256
    num_iters = cute.ceil_div(total_elems, threads_per_block)
    
    shared_sum = cute.arch.shared_memory(float, 256)
    
    local_sum = 0.0
    for i in range(num_iters):
        idx = i * threads_per_block + tidx
        if idx < total_elems:
            hi = idx // W
            wi = idx % W
            local_sum += gA[batch_idx, channel_idx, hi, wi]
    
    shared_sum[tidx] = local_sum
    
    cute.arch.sync()
    for s in range(8):
        step = 256 >> (s + 1)
        if tidx < step:
            shared_sum[tidx] += shared_sum[tidx + step]
        cute.arch.sync()
    
    if tidx == 0:
        total_sum = shared_sum[0]
        gC[batch_idx, channel_idx, 0, 0] = total_sum * multiplier / num

@cute.jit
def multiply_and_pool_host(mA: cute.Tensor, multiplier: float, mC: cute.Tensor):
    batch = mA.shape[0]
    channels = mA.shape[1]
    H = mA.shape[2]
    W = mA.shape[3]
    num = H * W
    threads_per_block = 256
    grid_x = batch * channels
    
    multiply_and_pool_kernel(mA, multiplier, num, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1), shared_memory=256 * 4)

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, then fuses multiply by scalar and global average pooling into a single CuTe kernel.
    The second global average pooling is redundant (no-op after the first) and is omitted.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        A = x.contiguous().cuda()
        batch, channels, H, W = A.shape
        C = torch.empty((batch, channels, 1, 1), dtype=A.dtype, device=A.device)
        
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (A.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(multiply_and_pool_host, mA, self.multiplier, mC)
            self.compiled[key] = compiled
        
        compiled(mA, self.multiplier, mC)
        return C