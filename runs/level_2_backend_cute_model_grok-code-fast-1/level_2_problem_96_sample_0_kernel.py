import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def multiply_kernel(gA: cute.Tensor, scale: float, gC: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    thread_idx = bidx * bdim + tidx
    shape = gA.shape
    total = shape[0] * shape[1] * shape[2] * shape[3] * shape[4]
    if thread_idx < total:
        idx = thread_idx
        n = shape[4]
        w = idx % n
        idx //= n
        h = idx % shape[3]
        idx //= shape[3]
        d = idx % shape[2]
        idx //= shape[2]
        c = idx % shape[1]
        idx //= shape[1]
        b = idx
        gC[b, c, d, h, w] = gA[b, c, d, h, w] * scale

@cute.jit
def multiply_host(mA: cute.Tensor, scale: float, mC: cute.Tensor):
    total = mA.shape[0] * mA.shape[1] * mA.shape[2] * mA.shape[3] * mA.shape[4]
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    multiply_kernel(mA, scale, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def avg_clamp_kernel(gA: cute.Tensor, gC: cute.Tensor, min_val: float, max_val: float):
    bidx, cidx, _ = cute.arch.block_idx()
    d, h, w = gA.shape[2], gA.shape[3], gA.shape[4]
    sum_val = 0.0
    for i in range(d):
        for j in range(h):
            for k in range(w):
                sum_val += gA[bidx, cidx, i, j, k]
    num = d * h * w
    avg = sum_val / num
    gC[bidx, cidx, 0, 0, 0] = max(min_val, min(max_val, avg))

@cute.jit
def avg_clamp_host(mA: cute.Tensor, mC: cute.Tensor, min_val: float, max_val: float):
    batch, channels = mA.shape[0], mA.shape[1]
    avg_clamp_kernel(mA, mC, min_val, max_val).launch(grid=(batch, channels, 1), block=(1, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        weight = self.conv_transpose.weight.permute(1, 0, 2, 3, 4)
        self.conv_op = cutlass.ops.Conv3d(
            input_channels=in_channels,
            output_channels=out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=(stride, stride, stride),
            padding=(padding, padding, padding),
            mode='fprop'
        )
        self.conv_op.set_weight(weight)
        if self.conv_transpose.bias is not None:
            self.conv_op.set_bias(self.conv_transpose.bias)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.clamp_min = 0
        self.clamp_max = 1
        self.compiled_multiply = {}
        self.compiled_avg_clamp = {}

    def forward(self, x):
        x = x.contiguous().cuda()
        x = self.conv_op(x)
        M, C, D, H, W = x.shape
        x_mult = torch.empty_like(x)
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mC = from_dlpack(x_mult, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key = (x.dtype,)
        compiled_mult = self.compiled_multiply.get(key)
        if compiled_mult is None:
            compiled_mult = cute.compile(multiply_host, mA, self.scale, mC)
            self.compiled_multiply[key] = compiled_mult
        compiled_mult(mA, self.scale, mC)
        x = x_mult
        x = self.maxpool(x)
        batch, channels, d, h, w = x.shape
        x_out = torch.empty((batch, channels, 1, 1, 1), dtype=x.dtype, device=x.device)
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mC = from_dlpack(x_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        compiled_avg = self.compiled_avg_clamp.get(key)
        if compiled_avg is None:
            compiled_avg = cute.compile(avg_clamp_host, mA, mC, self.clamp_min, self.clamp_max)
            self.compiled_avg_clamp[key] = compiled_avg
        compiled_avg(mA, mC, self.clamp_min, self.clamp_max)
        return x_out