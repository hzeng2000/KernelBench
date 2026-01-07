import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_scale_leaky_gelu_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gM: cute.Tensor, gY: cute.Tensor,
    batch_size: int, in_c: int, in_h: int, in_w: int,
    out_c: int, out_h: int, out_w: int,
    kernel_size: int, stride: int, padding: int, negative_slope: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_id = tidz * bdimx * bdimy + tidy * bdimx + tidx
    block_id = bidz * cute.arch.grid_dim().x * cute.arch.grid_dim().y + bidy * cute.arch.grid_dim().x + bidx

    total_threads = bdimx * bdimy * bdimz
    total_blocks = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z

    for idx in range(block_id * total_threads + thread_id, batch_size * out_c * out_h * out_w, total_blocks * total_threads):
        n = idx // (out_c * out_h * out_w)
        rem = idx % (out_c * out_h * out_w)
        oc = rem // (out_h * out_w)
        rem = rem % (out_h * out_w)
        oh = rem // out_w
        ow = rem % out_w

        sum_val = 0.0
        for ic in range(in_c):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    ih = oh * stride - padding + kh
                    iw = ow * stride - padding + kw
                    if ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                        x_val = gX[n, ic, ih, iw]
                        w_val = gW[oc, ic, kh, kw]
                        sum_val += x_val * w_val

        if gB is not None:
            sum_val += gB[oc]

        multiplier = gM[oc, 0, 0]
        sum_val *= multiplier

        if sum_val < 0.0:
            sum_val *= negative_slope

        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = sum_val * sum_val * sum_val
        tanh_arg = 0.7978845608 * (sum_val + 0.044715 * x_cubed)
        tanh_val = cute.math.tanh(tanh_arg)
        gelu_val = 0.5 * sum_val * (1.0 + tanh_val)

        gY[n, oc, oh, ow] = gelu_val

@cute.jit
def conv_scale_leaky_gelu_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mM: cute.Tensor, mY: cute.Tensor,
    batch_size: int, in_c: int, in_h: int, in_w: int,
    out_c: int, out_h: int, out_w: int,
    kernel_size: int, stride: int, padding: int, negative_slope: float
):
    total_elements = batch_size * out_c * out_h * out_w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elements, threads_per_block)

    conv_scale_leaky_gelu_kernel(
        mX, mW, mB, mM, mY,
        batch_size, in_c, in_h, in_w,
        out_c, out_h, out_w,
        kernel_size, stride, padding, negative_slope
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.negative_slope = 0.01

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        self.compiled = {}

    def forward(self, x):
        batch_size, in_c, in_h, in_w = x.shape
        out_c = self.out_channels
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        multiplier = self.multiplier.contiguous().cuda()
        y = torch.empty(batch_size, out_c, out_h, out_w, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mM = from_dlpack(multiplier, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_scale_leaky_gelu_host,
                mX, mW, mB, mM, mY,
                batch_size, in_c, in_h, in_w,
                out_c, out_h, out_w,
                self.kernel_size, self.stride, self.padding, self.negative_slope
            )
            self.compiled[key] = compiled

        compiled(
            mX, mW, mB, mM, mY,
            batch_size, in_c, in_h, in_w,
            out_c, out_h, out_w,
            self.kernel_size, self.stride, self.padding, self.negative_slope
        )
        return y