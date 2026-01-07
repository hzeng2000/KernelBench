import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def min_clamp_kernel(gX: cute.Tensor, gY: cute.Tensor, min_val: float, max_val: float):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    n, c, d, h, w = gX.shape
    total_elems = n * c * d * h * w

    if thread_idx >= total_elems:
        return

    # Compute indices
    ni = thread_idx % w
    thread_idx //= w
    hi = thread_idx % h
    thread_idx //= h
    di = thread_idx % d
    thread_idx //= d
    ci = thread_idx % c
    thread_idx //= c
    bi = thread_idx

    x_val = gX[bi, ci, di, hi, ni]
    y_val = cute.min(x_val, min_val)
    gY[bi, ci, di, hi, ni] = cute.clamp(y_val, min_val, max_val)

@cute.jit
def min_clamp_host(mX: cute.Tensor, mY: cute.Tensor, min_val: float, max_val: float):
    N, C, D, H, W = mX.shape
    total_elems = N * C * D * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    min_clamp_kernel(mX, mY, min_val, max_val).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Group Normalization, fused minimum and clamp using custom CuTe kernel, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.min_value = min_value
        self.max_value = max_value
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        # Fused min and clamp
        N, C, D, H, W = x.shape
        x_cont = x.contiguous().cuda()
        y = torch.empty_like(x_cont)

        mX = from_dlpack(x_cont, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(min_clamp_host, mX, mY, self.min_value, self.max_value)
            self.compiled[key] = compiled

        compiled(mX, mY, self.min_value, self.max_value)
        x = y
        x = self.dropout(x)
        return x