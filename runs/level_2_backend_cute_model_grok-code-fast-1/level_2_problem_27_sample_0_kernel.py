import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def hardswish_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    thread_idx = bidx * bdim + tidx
    total = gX.numel()
    if thread_idx >= total:
        return
    val = gX.flat[thread_idx]
    gY.flat[thread_idx] = val * max(0.0, min(6.0, val + 3.0)) / 6.0

@cute.jit
def hardswish_host(mX: cute.Tensor, mY: cute.Tensor):
    total = mX.numel()
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    hardswish_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def mean_pool_kernel(gX: cute.Tensor, gY: cute.Tensor, S: int):
    b = cute.arch.block_idx().x
    c = cute.arch.block_idx().y
    tid = cute.arch.thread_idx().x
    shared_sum = cute.shared_memory(float, 256)
    sum_val = 0.0
    for i in range(tid, S, 256):
        sum_val += gX[b, c, i]
    shared_sum[tid] = sum_val
    cute.syncthreads()
    for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
        if tid < stride:
            shared_sum[tid] += shared_sum[tid + stride]
        cute.syncthreads()
    if tid == 0:
        gY[b, c] = shared_sum[0] / S

@cute.jit
def mean_pool_host(mX: cute.Tensor, mY: cute.Tensor, S: int):
    B, C, _ = mX.shape
    threads_per_block = 256
    grid = (B, C, 1)
    block = (threads_per_block, 1, 1)
    mean_pool_kernel(mX, mY, S).launch(grid=grid, block=block)

class ModelNew(nn.Module):
    """
    Optimized Model that performs:
    1. Conv3D
    2. HardSwish activation (custom kernel)
    3. GroupNorm  
    4. Mean pooling across spatial dimensions (custom kernel)
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.compiled = {'hardswish': {}, 'mean': {}}

    def forward(self, x):
        x = self.conv(x)                             # (B, C, D, H, W)
        x = x.contiguous().cuda()
        y_hardswish = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY_hardswish = from_dlpack(y_hardswish, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key_hardswish = x.dtype
        compiled_hardswish = self.compiled['hardswish'].get(key_hardswish)
        if compiled_hardswish is None:
            compiled_hardswish = cute.compile(hardswish_host, mX, mY_hardswish)
            self.compiled['hardswish'][key_hardswish] = compiled_hardswish
        compiled_hardswish(mX, mY_hardswish)
        x = y_hardswish
        x = self.group_norm(x)                       # Normalization over channels
        x = x.contiguous()
        B, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.view(B, C, S)
        y_mean = torch.empty((B, C), dtype=x.dtype, device=x.device)
        mX_flat = from_dlpack(x_flat, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mY_mean = from_dlpack(y_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key_mean = (x.dtype, S)
        compiled_mean = self.compiled['mean'].get(key_mean)
        if compiled_mean is None:
            compiled_mean = cute.compile(mean_pool_host, mX_flat, mY_mean, S)
            self.compiled['mean'][key_mean] = compiled_mean
        compiled_mean(mX_flat, mY_mean, S)
        return y_mean