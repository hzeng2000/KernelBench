import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def elementwise_div_kernel(gX: cute.Tensor, divisor: float, gY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    shape = gX.shape
    total = 1
    for s in shape:
        total *= s
    if thread_idx < total:
        w = shape[4]
        h = shape[3]
        d = shape[2]
        c = shape[1]
        b = shape[0]
        w_idx = thread_idx % w
        temp = thread_idx // w
        h_idx = temp % h
        temp //= h
        d_idx = temp % d
        temp //= d
        c_idx = temp % c
        b_idx = temp // c
        gY[b_idx, c_idx, d_idx, h_idx, w_idx] = gX[b_idx, c_idx, d_idx, h_idx, w_idx] / divisor

@cute.jit
def elementwise_div_host(mX: cute.Tensor, divisor: float, mY: cute.Tensor):
    shape = mX.shape
    total = 1
    for s in shape:
        total *= s
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    elementwise_div_kernel(mX, divisor, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def add_bias_kernel(gX: cute.Tensor, gBias: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    shape = gX.shape
    total = 1
    for s in shape:
        total *= s
    if thread_idx < total:
        w = shape[4]
        h = shape[3]
        d = shape[2]
        c = shape[1]
        b = shape[0]
        w_idx = thread_idx % w
        temp = thread_idx // w
        h_idx = temp % h
        temp //= h
        d_idx = temp % d
        temp //= d
        c_idx = temp % c
        b_idx = temp // c
        gC[b_idx, c_idx, d_idx, h_idx, w_idx] = gX[b_idx, c_idx, d_idx, h_idx, w_idx] + gBias[c_idx, 0, 0, 0]

@cute.jit
def add_bias_host(mX: cute.Tensor, mBias: cute.Tensor, mC: cute.Tensor):
    shape = mX.shape
    total = 1
    for s in shape:
        total *= s
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    add_bias_kernel(mX, mBias, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.compiled_div = {}
        self.compiled_add = {}

    def forward(self, x):
        x = self.conv(x)
        # Custom elementwise divide
        x = x.contiguous().cuda()
        y = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key = (x.dtype,)
        compiled = self.compiled_div.get(key)
        if compiled is None:
            compiled = cute.compile(elementwise_div_host, mX, self.divisor, mY)
            self.compiled_div[key] = compiled
        compiled(mX, self.divisor, mY)
        x = y
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        # Custom add bias
        x = x.contiguous().cuda()
        self.bias = self.bias.cuda()
        z = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mZ = from_dlpack(z, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key = (x.dtype,)
        compiled = self.compiled_add.get(key)
        if compiled is None:
            compiled = cute.compile(add_bias_host, mX, mBias, mZ)
            self.compiled_add[key] = compiled
        compiled(mX, mBias, mZ)
        x = z
        x = torch.sum(x, dim=self.sum_dim)
        return x