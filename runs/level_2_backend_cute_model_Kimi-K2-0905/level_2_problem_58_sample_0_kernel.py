import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose3d_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_c: int, in_d: int, in_h: int, in_w: int,
    out_c: int, out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    gdimx, gdimy, gdimz = cute.arch.grid_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_z < out_d and out_y < out_h and out_x < out_w:
        for n in range(batch_size):
            for oc in range(out_c):
                acc = 0.0
                for ic in range(in_c):
                    for kd in range(k_d):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                in_z = (out_z + pad_d - kd) // stride_d
                                in_y = (out_y + pad_h - kh) // stride_h
                                in_x = (out_x + pad_w - kw) // stride_w
                                if (out_z + pad_d - kd) % stride_d == 0 and \
                                   (out_y + pad_h - kh) % stride_h == 0 and \
                                   (out_x + pad_w - kw) % stride_w == 0 and \
                                   in_z >= 0 and in_z < in_d and \
                                   in_y >= 0 and in_y < in_h and \
                                   in_x >= 0 and in_x < in_w:
                                    acc += gInput[n, ic, in_z, in_y, in_x] * gWeight[ic, oc, kd, kh, kw]
                gOutput[n, oc, out_z, out_y, out_x] = acc

@cute.kernel
def logsumexp_kernel(gInput: cute.Tensor, gOutput: cute.Tensor, batch_size: int, channels: int, d: int, h: int, w: int):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    n = bidx
    z = bidy * bdimy + tidy
    y = bidz * bdimz + tidz
    x = tidx

    if n < batch_size and z < d and y < h and x < w:
        max_val = -1e38
        for c in range(channels):
            val = gInput[n, c, z, y, x]
            if val > max_val:
                max_val = val
        
        sum_exp = 0.0
        for c in range(channels):
            sum_exp += cute.exp(gInput[n, c, z, y, x] - max_val)
        
        gOutput[n, 0, z, y, x] = max_val + cute.log(sum_exp)

@cute.kernel
def hardswish_sub_clamp_kernel(
    gInput: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, d: int, h: int, w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    n = bidx
    z = bidy * bdimy + tidy
    y = bidz * bdimz + tidz
    x = tidx

    if n < batch_size and z < d and y < h and x < w:
        val = gInput[n, 0, z, y, x]
        sigmoid_val = cute.sigmoid(val + 3.0)
        hardswish_val = val * sigmoid_val / 6.0
        sub_val = hardswish_val - gBias[0, 0, 0, 0]
        clamped_val = cute.max(-1.0, cute.min(1.0, sub_val))
        gOutput[n, 0, z, y, x] = clamped_val

@cute.jit
def conv_transpose3d_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_c: int, in_d: int, in_h: int, in_w: int,
    out_c: int, out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    threads_per_block = 256
    blocks_x = cute.ceil_div(out_w, 8)
    blocks_y = cute.ceil_div(out_h, 8)
    blocks_z = cute.ceil_div(out_d, 8)
    grid = (blocks_x, blocks_y, blocks_z)
    block = (8, 8, 4)
    conv_transpose3d_kernel(
        mInput, mWeight, mOutput,
        batch_size, in_c, in_d, in_h, in_w,
        out_c, out_d, out_h, out_w,
        k_d, k_h, k_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    ).launch(grid=grid, block=block)

@cute.jit
def logsumexp_host(mInput: cute.Tensor, mOutput: cute.Tensor, batch_size: int, channels: int, d: int, h: int, w: int):
    threads_per_block = 256
    blocks_x = cute.ceil_div(w, 8)
    blocks_y = cute.ceil_div(h, 8)
    blocks_z = cute.ceil_div(d, 8)
    grid = (batch_size, blocks_y, blocks_z)
    block = (8, 8, 8)
    logsumexp_kernel(mInput, mOutput, batch_size, channels, d, h, w).launch(grid=grid, block=block)

@cute.jit
def hardswish_sub_clamp_host(mInput: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor, batch_size: int, d: int, h: int, w: int):
    threads_per_block = 256
    blocks_x = cute.ceil_div(w, 8)
    blocks_y = cute.ceil_div(h, 8)
    blocks_z = cute.ceil_div(d, 8)
    grid = (batch_size, blocks_y, blocks_z)
    block = (8, 8, 8)
    hardswish_sub_clamp_kernel(mInput, mBias, mOutput, batch_size, d, h, w).launch(grid=grid, block=block)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.compiled = {}

    def forward(self, x):
        batch_size, _, in_d, in_h, in_w = x.shape
        out_d = (in_d - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        out_h = (in_h - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
        out_w = (in_w - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2]

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()

        conv_out = torch.empty(batch_size, self.out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        logsumexp_out = torch.empty(batch_size, 1, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        final_out = torch.empty(batch_size, 1, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mLogsumexpOut = from_dlpack(logsumexp_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mFinalOut = from_dlpack(final_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = {
                'conv': cute.compile(conv_transpose3d_host, mInput, mWeight, mConvOut,
                                     batch_size, self.in_channels, in_d, in_h, in_w,
                                     self.out_channels, out_d, out_h, out_w,
                                     self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                                     self.stride[0], self.stride[1], self.stride[2],
                                     self.padding[0], self.padding[1], self.padding[2]),
                'logsumexp': cute.compile(logsumexp_host, mConvOut, mLogsumexpOut, batch_size, self.out_channels, out_d, out_h, out_w),
                'hardswish': cute.compile(hardswish_sub_clamp_host, mLogsumexpOut, mBias, mFinalOut, batch_size, out_d, out_h, out_w)
            }
            self.compiled[key] = compiled

        compiled['conv'](mInput, mWeight, mConvOut,
                         batch_size, self.in_channels, in_d, in_h, in_w,
                         self.out_channels, out_d, out_h, out_w,
                         self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                         self.stride[0], self.stride[1], self.stride[2],
                         self.padding[0], self.padding[1], self.padding[2])
        compiled['logsumexp'](mConvOut, mLogsumexpOut, batch_size, self.out_channels, out_d, out_h, out_w)
        compiled['hardswish'](mLogsumexpOut, mBias, mFinalOut, batch_size, out_d, out_h, out_w)

        return final_out