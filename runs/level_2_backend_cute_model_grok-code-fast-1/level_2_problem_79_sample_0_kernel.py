import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_post_kernel(gX: cute.Tensor, gMult: cute.Tensor, gY: cute.Tensor, clamp_min: float, clamp_max: float):
    b, c, d, h, w = gX.shape
    bidx = cute.arch.block_idx(0)
    tidx = cute.arch.thread_idx(0)
    total_spatial = d * h * w
    batch_idx = bidx // total_spatial
    spatial_idx = bidx % total_spatial
    depth_idx = spatial_idx // (h * w)
    hw_idx = spatial_idx % (h * w)
    height_idx = hw_idx // w
    width_idx = hw_idx % w
    if tidx < c:
        val = gX[batch_idx, tidx, depth_idx, height_idx, width_idx] * gMult[tidx, 0, 0, 0]
        val = max(clamp_min, min(clamp_max, val))
        val = val * gMult[tidx, 0, 0, 0]
        shared = cute.shared_memory(float, (32,))
        shared[tidx] = val
        cute.syncthreads()
        step = 1
        while step < c:
            if tidx % (2 * step) == 0 and tidx + step < c:
                shared[tidx] = max(shared[tidx], shared[tidx + step])
            step *= 2
            cute.syncthreads()
        if tidx == 0:
            gY[batch_idx, depth_idx, height_idx, width_idx] = shared[0]

@cute.jit
def fused_post_host(mX: cute.Tensor, mMult: cute.Tensor, mY: cute.Tensor, clamp_min: float, clamp_max: float):
    b, c, d, h, w = mX.shape
    grid_x = b * d * h * w
    threads_per_block = 32
    fused_post_kernel(mX, mMult, mY, clamp_min, clamp_max).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by instance normalization, then a fused kernel for multiplication, clamping, multiplication, and max reduction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        b, c, d, h, w = x.shape
        x = x.contiguous().cuda()
        mult = self.multiplier.contiguous().cuda()
        y = torch.empty((b, d, h, w), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mMult = from_dlpack(mult, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_host, mX, mMult, mY, self.clamp_min, self.clamp_max)
            self.compiled[key] = compiled
        compiled(mX, mMult, mY, self.clamp_min, self.clamp_max)
        return y