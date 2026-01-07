import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def scale_min_kernel(gX: cute.Tensor, scale: float, gY: cute.Tensor):
    bidx = cute.arch.block_idx(0)
    B, C, H, W = gX.shape
    total_spatial = B * H * W
    spatial_idx = bidx
    if spatial_idx >= total_spatial:
        return
    b = spatial_idx // (H * W)
    hw = spatial_idx % (H * W)
    h = hw // W
    w = hw % W
    tidx = cute.arch.thread_idx(0)
    if tidx >= C:
        return
    val = gX[b, tidx, h, w] * scale
    sX = cute.shared_tensor(cute.float32, (C,))
    sX[tidx] = val
    cute.sync()
    min_val = cute.reduce(sX, cute.reduce_op.min)
    if tidx == 0:
        gY[b, 0, h, w] = min_val

@cute.jit
def scale_min_host(mX: cute.Tensor, scale: float, mY: cute.Tensor):
    B, C, H, W = mX.shape
    total_spatial = B * H * W
    block = (C, 1, 1)
    grid = (total_spatial, 1, 1)
    scale_min_kernel(mX, scale, mY).launch(grid=grid, block=block)

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = cutlass.ops.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, activation=None, dtype=torch.float32)
        self.scale_factor = scale_factor
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        y = torch.empty((B, 1, H, W), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(scale_min_host, mX, self.scale_factor, mY)
            self.compiled[key] = compiled
        compiled(mX, self.scale_factor, mY)
        return y