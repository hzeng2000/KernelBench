import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_conv3d_scale_tanh_mul_sigmoid_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gScale: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, in_c: int, out_c: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    gdimx, gdimy, gdimz = cute.arch.grid_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_z >= out_d or out_y >= out_h or out_x >= out_w:
        return

    for n in range(batch_size):
        for oc in range(out_c):
            acc = 0.0
            for ic in range(in_c):
                for kd in range(k_d):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            in_z = out_z + kd
                            in_y = out_y + kh
                            in_x = out_x + kw
                            if in_z < in_d and in_y < in_h and in_x < in_w:
                                x_val = gX[n, ic, in_z, in_y, in_x]
                                w_val = gW[oc, ic, kd, kh, kw]
                                acc += x_val * w_val
            acc += gB[oc]
            acc *= gScale[oc, 0, 0, 0]
            acc = cute.math.tanh(acc)
            acc *= gBias[oc, 0, 0, 0]
            acc = 1.0 / (1.0 + cute.math.exp(-acc))
            gOut[n, oc, out_z, out_y, out_x] = acc

@cute.jit
def fused_conv3d_scale_tanh_mul_sigmoid_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mScale: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, in_c: int, out_c: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(out_w, threads_per_block)
    grid_y = cute.ceil_div(out_h, threads_per_block)
    grid_z = cute.ceil_div(out_d, threads_per_block)

    fused_conv3d_scale_tanh_mul_sigmoid_kernel(
        mX, mW, mB, mScale, mBias, mOut,
        batch_size, in_c, out_c,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        k_d, k_h, k_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        in_c = x.shape[1]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        out_c = self.conv.out_channels
        k_d, k_h, k_w = self.conv.kernel_size[0], self.conv.kernel_size[1], self.conv.kernel_size[2]

        out_d = in_d - k_d + 1
        out_h = in_h - k_h + 1
        out_w = in_w - k_w + 1

        x = x.contiguous().cuda()
        W = self.conv.weight.data.contiguous().cuda()
        B = self.conv.bias.data.contiguous().cuda()
        scale = self.scaling_factor.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty(batch_size, out_c, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mScale = from_dlpack(scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_conv3d_scale_tanh_mul_sigmoid_host,
                mX, mW, mB, mScale, mBias, mOut,
                batch_size, in_c, out_c,
                in_d, in_h, in_w,
                out_d, out_h, out_w,
                k_d, k_h, k_w
            )
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mScale, mBias, mOut,
                 batch_size, in_c, out_c,
                 in_d, in_h, in_w,
                 out_d, out_h, out_w,
                 k_d, k_h, k_w)
        return out