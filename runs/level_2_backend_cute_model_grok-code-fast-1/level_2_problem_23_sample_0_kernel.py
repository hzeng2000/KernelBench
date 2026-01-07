import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def sum_partial_kernel(gX: cute.Tensor, gPartial: cute.Tensor, total_elems, num_blocks, out_channels, D, H, W):
    bidx = cute.arch.block_idx().x
    n = bidx // num_blocks
    block_id = bidx % num_blocks
    tid = cute.arch.thread_idx().x
    idx = block_id * 256 + tid
    if idx < total_elems:
        c = idx // (D * H * W)
        d = (idx % (D * H * W)) // (H * W)
        h = (idx % (H * W)) // W
        w = idx % W
        val = gX[n, c, d, h, w]
        shared = cute.shared_memory(float, 256)
        shared[tid] = val
        cute.sync()
        s = 1
        while s < 256:
            if tid % (2 * s) == 0 and tid + s < 256:
                shared[tid] += shared[tid + s]
            s *= 2
            cute.sync()
        if tid == 0:
            gPartial[n, block_id] = shared[0]

@cute.kernel
def reduce_partial_kernel(gPartial: cute.Tensor, gY: cute.Tensor, num_blocks, total_elems):
    n = cute.arch.block_idx().x
    tid = cute.arch.thread_idx().x
    if tid == 0:
        sum_val = 0.0
        for i in range(num_blocks):
            sum_val += gPartial[n, i]
        gY[n, 0] = sum_val / total_elems

@cute.jit
def mean_host(mX, mPartial, mY, total_elems, num_blocks, out_channels, D, H, W):
    sum_partial_kernel(mX, mPartial, total_elems, num_blocks, out_channels, D, H, W).launch(grid=(batch_size * num_blocks, 1, 1), block=(256, 1, 1))
    reduce_partial_kernel(mPartial, mY, num_blocks, total_elems).launch(grid=(batch_size, 1, 1), block=(1, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = cutlass.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        batch_size, out_channels, D, H, W = x.shape
        total_elems = out_channels * D * H * W
        threads_per_block = 256
        num_blocks = cute.ceil_div(total_elems, threads_per_block)
        partial = torch.empty((batch_size, num_blocks), dtype=x.dtype, device=x.device)
        y = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPartial = from_dlpack(partial, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(mean_host, mX, mPartial, mY, total_elems, num_blocks, out_channels, D, H, W)
            self.compiled[key] = compiled
        compiled(mX, mPartial, mY, total_elems, num_blocks, out_channels, D, H, W)
        return y