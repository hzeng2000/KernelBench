import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_softmax_kernel(gX: cute.Tensor, clamp_min: float, clamp_max: float, gScale: cute.Tensor, gOut: cute.Tensor):
    b, c, d, h, w = gX.shape
    spatial_size = d * h * w
    block_idx = cute.arch.block_idx(0)
    b_idx = block_idx // c
    c_idx = block_idx % c
    thread_idx = cute.arch.thread_idx(0)
    block_dim = cute.arch.block_dim(0)
    num_tiles = cute.ceil_div(spatial_size, block_dim)
    
    shared_max = cute.shared_memory(float, (block_dim,))
    shared_sum = cute.shared_memory(float, (block_dim,))
    
    # Compute local max
    local_max = -float('inf')
    for tile in range(num_tiles):
        idx = tile * block_dim + thread_idx
        if idx < spatial_size:
            d_idx = idx // (h * w)
            temp = idx % (h * w)
            h_idx = temp // w
            w_idx = temp % w
            val = gX[b_idx, c_idx, d_idx, h_idx, w_idx]
            clamped = cute.max(cute.min(val, clamp_max), clamp_min)
            local_max = cute.max(local_max, clamped)
    shared_max[thread_idx] = local_max
    cute.sync()
    
    # Reduce max
    s = block_dim // 2
    while s > 0:
        if thread_idx < s:
            shared_max[thread_idx] = cute.max(shared_max[thread_idx], shared_max[thread_idx + s])
        cute.sync()
        s //= 2
    global_max = shared_max[0]
    
    # Compute local sum
    local_sum = 0.0
    for tile in range(num_tiles):
        idx = tile * block_dim + thread_idx
        if idx < spatial_size:
            d_idx = idx // (h * w)
            temp = idx % (h * w)
            h_idx = temp // w
            w_idx = temp % w
            val = gX[b_idx, c_idx, d_idx, h_idx, w_idx]
            clamped = cute.max(cute.min(val, clamp_max), clamp_min)
            local_sum += cute.exp(clamped - global_max)
    shared_sum[thread_idx] = local_sum
    cute.sync()
    
    # Reduce sum
    s = block_dim // 2
    while s > 0:
        if thread_idx < s:
            shared_sum[thread_idx] += shared_sum[thread_idx + s]
        cute.sync()
        s //= 2
    global_sum = shared_sum[0]
    
    # Apply softmax and scale
    scale_val = gScale[0, c_idx, 0, 0, 0]
    for tile in range(num_tiles):
        idx = tile * block_dim + thread_idx
        if idx < spatial_size:
            d_idx = idx // (h * w)
            temp = idx % (h * w)
            h_idx = temp // w
            w_idx = temp % w
            val = gX[b_idx, c_idx, d_idx, h_idx, w_idx]
            clamped = cute.max(cute.min(val, clamp_max), clamp_min)
            softmax_val = cute.exp(clamped - global_max) / global_sum
            gOut[b_idx, c_idx, d_idx, h_idx, w_idx] = softmax_val * scale_val

@cute.jit
def fused_softmax_host(mX: cute.Tensor, clamp_min: float, clamp_max: float, mScale: cute.Tensor, mOut: cute.Tensor):
    b, c, d, h, w = mX.shape
    threads_per_block = 1024
    grid_x = b * c
    fused_softmax_kernel(mX, clamp_min, clamp_max, mScale, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.compiled = {}

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        b, c, d, h, w = x.shape
        x = x.contiguous().cuda()
        out = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mScale = from_dlpack(self.scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_softmax_host, mX, self.clamp_min, self.clamp_max, mScale, mOut)
            self.compiled[key] = compiled
        compiled(mX, self.clamp_min, self.clamp_max, mScale, mOut)
        return out