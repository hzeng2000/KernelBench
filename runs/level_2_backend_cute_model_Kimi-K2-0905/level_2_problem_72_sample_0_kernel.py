import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose3d_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_c: int, in_d: int, in_h: int, in_w: int,
    out_c: int, out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

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
                                in_d_idx = (out_z + pad_d - kd) // stride_d
                                in_h_idx = (out_y + pad_h - kh) // stride_h
                                in_w_idx = (out_x + pad_w - kw) // stride_w
                                if (out_z + pad_d - kd) % stride_d == 0 and \
                                   (out_y + pad_h - kh) % stride_h == 0 and \
                                   (out_x + pad_w - kw) % stride_w == 0 and \
                                   in_d_idx >= 0 and in_d_idx < in_d and \
                                   in_h_idx >= 0 and in_h_idx < in_h and \
                                   in_w_idx >= 0 and in_w_idx < in_w:
                                    acc += gInput[n, ic, in_d_idx, in_h_idx, in_w_idx] * \
                                           gWeight[oc, ic, kd, kh, kw]
                gOutput[n, oc, out_z, out_y, out_x] = acc + gBias[oc, 0, 0, 0]

@cute.kernel
def batch_norm3d_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor,
    gGamma: cute.Tensor, gBeta: cute.Tensor, batch_size: int, channels: int, depth: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    w = bidx * bdimx + tidx
    h = bidy * bdimy + tidy
    d = bidz * bdimz + tidz

    if w < width and h < height and d < depth:
        for n in range(batch_size):
            for c in range(channels):
                val = gInput[n, c, d, h, w]
                mean = gMean[c]
                var = gVar[c]
                gamma = gGamma[c]
                beta = gBeta[c]
                gOutput[n, c, d, h, w] = gamma * (val - mean) / cute.sqrt(var + 1e-5) + beta

@cute.kernel
def avg_pool3d_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor, batch_size: int, channels: int,
    in_d: int, in_h: int, in_w: int, out_d: int, out_h: int, out_w: int, kernel_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_w_idx = bidx * bdimx + tidx
    out_h_idx = bidy * bdimy + tidy
    out_d_idx = bidz * bdimz + tidz

    if out_w_idx < out_w and out_h_idx < out_h and out_d_idx < out_d:
        for n in range(batch_size):
            for c in range(channels):
                acc = 0.0
                count = 0
                for kd in range(kernel_size):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            in_d_pos = out_d_idx * kernel_size + kd
                            in_h_pos = out_h_idx * kernel_size + kh
                            in_w_pos = out_w_idx * kernel_size + kw
                            if in_d_pos < in_d and in_h_pos < in_h and in_w_pos < in_w:
                                acc += gInput[n, c, in_d_pos, in_h_pos, in_w_pos]
                                count += 1
                gOutput[n, c, out_d_idx, out_h_idx, out_w_idx] = acc / count

@cute.jit
def conv_transpose3d_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_c: int, in_d: int, in_h: int, in_w: int,
    out_c: int, out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(out_w, threads_per_block)
    grid_y = cute.ceil_div(out_h, threads_per_block)
    grid_z = cute.ceil_div(out_d, threads_per_block)
    conv_transpose3d_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_c, in_d, in_h, in_w,
        out_c, out_d, out_h, out_w,
        k_d, k_h, k_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

@cute.jit
def batch_norm3d_host(
    mInput: cute.Tensor, mOutput: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor,
    mGamma: cute.Tensor, mBeta: cute.Tensor, batch_size: int, channels: int, depth: int, height: int, width: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(width, threads_per_block)
    grid_y = cute.ceil_div(height, threads_per_block)
    grid_z = cute.ceil_div(depth, threads_per_block)
    batch_norm3d_kernel(
        mInput, mOutput, mMean, mVar, mGamma, mBeta, batch_size, channels, depth, height, width
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

@cute.jit
def avg_pool3d_host(
    mInput: cute.Tensor, mOutput: cute.Tensor, batch_size: int, channels: int,
    in_d: int, in_h: int, in_w: int, out_d: int, out_h: int, out_w: int, kernel_size: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(out_w, threads_per_block)
    grid_y = cute.ceil_div(out_h, threads_per_block)
    grid_z = cute.ceil_div(out_d, threads_per_block)
    avg_pool3d_kernel(
        mInput, mOutput, batch_size, channels, in_d, in_h, in_w, out_d, out_h, out_w, kernel_size
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.compiled = {}

    def forward(self, x):
        batch_size = x.size(0)
        in_d, in_h, in_w = x.size(2), x.size(3), x.size(4)
        out_d = (in_d - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()

        conv_out = torch.empty(batch_size, self.out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_transpose3d_host, mInput, mWeight, mBias, mConvOut,
                                    batch_size, self.in_channels, in_d, in_h, in_w,
                                    self.out_channels, out_d, out_h, out_w,
                                    self.kernel_size, self.kernel_size, self.kernel_size,
                                    self.stride, self.stride, self.stride,
                                    self.padding, self.padding, self.padding)
            self.compiled[key] = compiled

        compiled(mInput, mWeight, mBias, mConvOut,
                 batch_size, self.in_channels, in_d, in_h, in_w,
                 self.out_channels, out_d, out_h, out_w,
                 self.kernel_size, self.kernel_size, self.kernel_size,
                 self.stride, self.stride, self.stride,
                 self.padding, self.padding, self.padding)

        bn_out = torch.empty_like(conv_out)
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBnOut = from_dlpack(bn_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mean = self.running_mean.contiguous().cuda()
        var = self.running_var.contiguous().cuda()
        gamma = self.gamma.contiguous().cuda()
        beta = self.beta.contiguous().cuda()
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGamma = from_dlpack(gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        compiled_bn = cute.compile(batch_norm3d_host, mConvOut, mBnOut, mMean, mVar, mGamma, mBeta,
                                   batch_size, self.out_channels, out_d, out_h, out_w)
        compiled_bn(mConvOut, mBnOut, mMean, mVar, mGamma, mBeta,
                    batch_size, self.out_channels, out_d, out_h, out_w)

        pool1_out = torch.empty(batch_size, self.out_channels, out_d // 2, out_h // 2, out_w // 2, dtype=x.dtype, device=x.device)
        mBnOut = from_dlpack(bn_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPool1Out = from_dlpack(pool1_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        compiled_pool1 = cute.compile(avg_pool3d_host, mBnOut, mPool1Out,
                                      batch_size, self.out_channels, out_d, out_h, out_w,
                                      out_d // 2, out_h // 2, out_w // 2, 2)
        compiled_pool1(mBnOut, mPool1Out,
                       batch_size, self.out_channels, out_d, out_h, out_w,
                       out_d // 2, out_h // 2, out_w // 2, 2)

        pool2_out = torch.empty(batch_size, self.out_channels, (out_d // 2) // 2, (out_h // 2) // 2, (out_w // 2) // 2, dtype=x.dtype, device=x.device)
        mPool1Out = from_dlpack(pool1_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPool2Out = from_dlpack(pool2_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        compiled_pool2 = cute.compile(avg_pool3d_host, mPool1Out, mPool2Out,
                                      batch_size, self.out_channels, out_d // 2, out_h // 2, out_w // 2,
                                      (out_d // 2) // 2, (out_h // 2) // 2, (out_w // 2) // 2, 2)
        compiled_pool2(mPool1Out, mPool2Out,
                       batch_size, self.out_channels, out_d // 2, out_h // 2, out_w // 2,
                       (out_d // 2) // 2, (out_h // 2) // 2, (out_w // 2) // 2, 2)

        return pool2_out