import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_transpose3d_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_c: int, in_d: int, in_h: int, in_w: int,
    out_c: int, out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int, out_pad_d: int, out_pad_h: int, out_pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_x < out_w and out_y < out_h and out_z < out_d:
        for n in range(batch_size):
            for oc in range(out_c):
                acc = 0.0
                for ic in range(in_c):
                    for kd in range(k_d):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                in_d_idx = (out_z + pad_d - kd * stride_d) // stride_d
                                in_h_idx = (out_y + pad_h - kh * stride_h) // stride_h
                                in_w_idx = (out_x + pad_w - kw * stride_w) // stride_w
                                if (in_d_idx >= 0 and in_d_idx < in_d and
                                    in_h_idx >= 0 and in_h_idx < in_h and
                                    in_w_idx >= 0 and in_w_idx < in_w and
                                    (out_z + pad_d - kd * stride_d) % stride_d == 0 and
                                    (out_y + pad_h - kh * stride_h) % stride_h == 0 and
                                    (out_x + pad_w - kw * stride_w) % stride_w == 0):
                                    acc += gInput[n, ic, in_d_idx, in_h_idx, in_w_idx] * gWeight[oc, ic, kd, kh, kw]
                gOutput[n, oc, out_z, out_y, out_x] = acc

@cute.kernel
def leaky_relu_kernel(gX: cute.Tensor, gY: cute.Tensor, negative_slope: float):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    idx = bidx * bdim + tidx
    if idx < gX.numel():
        val = gX[idx]
        gY[idx] = val if val > 0.0 else val * negative_slope

@cute.kernel
def multiply_kernel(gX: cute.Tensor, gM: cute.Tensor, gY: cute.Tensor, c: int, d: int, h: int, w: int):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    idx = bidx * bdim + tidx
    if idx < gX.numel():
        n = idx // (c * d * h * w)
        rem = idx % (c * d * h * w)
        oc = rem // (d * h * w)
        gY[idx] = gX[idx] * gM[oc, 0, 0, 0]

@cute.kernel
def max_pool3d_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, channels: int, in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int, k_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_x < out_w and out_y < out_h and out_z < out_d:
        for n in range(batch_size):
            for c in range(channels):
                max_val = -1e38
                for kd in range(k_size):
                    for kh in range(k_size):
                        for kw in range(k_size):
                            in_d_idx = out_z * k_size + kd
                            in_h_idx = out_y * k_size + kh
                            in_w_idx = out_x * k_size + kw
                            if in_d_idx < in_d and in_h_idx < in_h and in_w_idx < in_w:
                                val = gInput[n, c, in_d_idx, in_h_idx, in_w_idx]
                                if val > max_val:
                                    max_val = val
                gOutput[n, c, out_z, out_y, out_x] = max_val

@cute.jit
def conv_transpose3d_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_c: int, in_d: int, in_h: int, in_w: int,
    out_c: int, out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int, out_pad_d: int, out_pad_h: int, out_pad_w: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(out_w, 8)
    grid_y = cute.ceil_div(out_h, 8)
    grid_z = cute.ceil_div(out_d, 8)
    conv_transpose3d_kernel(
        mInput, mWeight, mOutput,
        batch_size, in_c, in_d, in_h, in_w,
        out_c, out_d, out_h, out_w,
        k_d, k_h, k_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(8, 8, 8))

@cute.jit
def leaky_relu_host(mX: cute.Tensor, mY: cute.Tensor, negative_slope: float):
    threads_per_block = 256
    total_elems = mX.numel()
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    leaky_relu_kernel(mX, mY, negative_slope).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.jit
def multiply_host(mX: cute.Tensor, mM: cute.Tensor, mY: cute.Tensor, c: int, d: int, h: int, w: int):
    threads_per_block = 256
    total_elems = mX.numel()
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    multiply_kernel(mX, mM, mY, c, d, h, w).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.jit
def max_pool3d_host(
    mInput: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, channels: int, in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int, k_size: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(out_w, 8)
    grid_y = cute.ceil_div(out_h, 8)
    grid_z = cute.ceil_div(out_d, 8)
    max_pool3d_kernel(
        mInput, mOutput,
        batch_size, channels, in_d, in_h, in_w,
        out_d, out_h, out_w, k_size
    ).launch(grid=(grid_x, grid_y, grid_z), block=(8, 8, 8))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        out_d = (in_d - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        conv_out = torch.empty(batch_size, self.out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_transpose3d_host, mInput, mWeight, mConvOut,
                                    batch_size, self.in_channels, in_d, in_h, in_w,
                                    self.out_channels, out_d, out_h, out_w,
                                    self.kernel_size, self.kernel_size, self.kernel_size,
                                    self.stride, self.stride, self.stride,
                                    self.padding, self.padding, self.padding,
                                    self.output_padding, self.output_padding, self.output_padding)
            self.compiled[key] = compiled
        compiled(mInput, mWeight, mConvOut,
                 batch_size, self.in_channels, in_d, in_h, in_w,
                 self.out_channels, out_d, out_h, out_w,
                 self.kernel_size, self.kernel_size, self.kernel_size,
                 self.stride, self.stride, self.stride,
                 self.padding, self.padding, self.padding,
                 self.output_padding, self.output_padding, self.output_padding)

        leaky1_out = torch.empty_like(conv_out)
        mConvOut = from_dlpack(conv_out, assumed_align=16).flatten()
        mLeaky1Out = from_dlpack(leaky1_out, assumed_align=16).flatten()
        compiled_leaky1 = cute.compile(leaky_relu_host, mConvOut, mLeaky1Out, 0.2)
        compiled_leaky1(mConvOut, mLeaky1Out, 0.2)

        multiplier = self.multiplier.contiguous().cuda()
        mult_out = torch.empty_like(leaky1_out)
        mLeaky1Out = from_dlpack(leaky1_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mMult = from_dlpack(multiplier, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mMultOut = from_dlpack(mult_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        compiled_mult = cute.compile(multiply_host, mLeaky1Out, mMult, mMultOut,
                                     self.out_channels, out_d, out_h, out_w)
        compiled_mult(mLeaky1Out, mMult, mMultOut, self.out_channels, out_d, out_h, out_w)

        leaky2_out = torch.empty_like(mult_out)
        mMultOut = from_dlpack(mult_out, assumed_align=16).flatten()
        mLeaky2Out = from_dlpack(leaky2_out, assumed_align=16).flatten()
        compiled_leaky2 = cute.compile(leaky_relu_host, mMultOut, mLeaky2Out, 0.2)
        compiled_leaky2(mMultOut, mLeaky2Out, 0.2)

        pool_out_d = out_d // 2
        pool_out_h = out_h // 2
        pool_out_w = out_w // 2
        pool_out = torch.empty(batch_size, self.out_channels, pool_out_d, pool_out_h, pool_out_w,
                               dtype=x.dtype, device=x.device)
        mLeaky2Out = from_dlpack(leaky2_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPoolOut = from_dlpack(pool_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        compiled_pool = cute.compile(max_pool3d_host, mLeaky2Out, mPoolOut,
                                     batch_size, self.out_channels, out_d, out_h, out_w,
                                     pool_out_d, pool_out_h, pool_out_w, 2)
        compiled_pool(mLeaky2Out, mPoolOut,
                      batch_size, self.out_channels, out_d, out_h, out_w,
                      pool_out_d, pool_out_h, pool_out_w, 2)

        return pool_out