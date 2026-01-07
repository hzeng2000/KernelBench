import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def swish_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx

    B, C, D, H, W = gX.shape
    total = B * C * D * H * W
    if thread_idx >= total:
        return

    b = thread_idx // (C * D * H * W)
    rem = thread_idx % (C * D * H * W)
    c = rem // (D * H * W)
    rem2 = rem % (D * H * W)
    d = rem2 // (H * W)
    rem3 = rem2 % (H * W)
    h = rem3 // W
    w = rem3 % W

    x_val = gX[b, c, d, h, w]
    sig = 1 / (1 + cute.exp(-x_val))
    gY[b, c, d, h, w] = x_val * sig

@cute.jit
def swish_host(mX: cute.Tensor, mY: cute.Tensor):
    B, C, D, H, W = mX.shape
    total = B * C * D * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    swish_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def hardswish_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx

    B, C, D, H, W = gX.shape
    total = B * C * D * H * W
    if thread_idx >= total:
        return

    b = thread_idx // (C * D * H * W)
    rem = thread_idx % (C * D * H * W)
    c = rem // (D * H * W)
    rem2 = rem % (D * H * W)
    d = rem2 // (H * W)
    rem3 = rem2 % (H * W)
    h = rem3 // W
    w = rem3 % W

    x_val = gX[b, c, d, h, w]
    val = (x_val + 3) / 6
    clamp_val = cute.max(0, cute.min(1, val))
    gY[b, c, d, h, w] = x_val * clamp_val

@cute.jit
def hardswish_host(mX: cute.Tensor, mY: cute.Tensor):
    B, C, D, H, W = mX.shape
    total = B * C * D * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    hardswish_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies custom Swish activation, 
    group normalization, and then custom HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        # Custom Swish
        x = x.contiguous().cuda()
        x_swish = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY_swish = from_dlpack(x_swish, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key_swish = ('swish', x.dtype)
        compiled_swish = self.compiled.get(key_swish)
        if compiled_swish is None:
            compiled_swish = cute.compile(swish_host, mX, mY_swish)
            self.compiled[key_swish] = compiled_swish
        compiled_swish(mX, mY_swish)
        x = x_swish
        x = self.group_norm(x)
        # Custom HardSwish
        x = x.contiguous().cuda()
        x_hardswish = torch.empty_like(x)
        mX_hard = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY_hard = from_dlpack(x_hardswish, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key_hardswish = ('hardswish', x.dtype)
        compiled_hardswish = self.compiled.get(key_hardswish)
        if compiled_hardswish is None:
            compiled_hardswish = cute.compile(hardswish_host, mX_hard, mY_hard)
            self.compiled[key_hardswish] = compiled_hardswish
        compiled_hardswish(mX_hard, mY_hard)
        return x_hardswish