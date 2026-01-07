import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_transpose_add_hardswish_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gAdd: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, out_c: int, out_d: int, out_h: int, out_w: int,
    in_c: int, in_d: int, in_h: int, in_w: int,
    k_d: int, k_h: int, k_w: int, stride: int, pad: int, out_pad: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    bid = bidz * cute.arch.grid_dim_x() * cute.arch.grid_dim_y() + bidy * cute.arch.grid_dim_x() + bidx
    idx = bid * (bdimx * bdimy * bdimz) + tid

    total = batch_size * out_c * out_d * out_h * out_w
    if idx >= total:
        return

    n = idx // (out_c * out_d * out_h * out_w)
    rem = idx % (out_c * out_d * out_h * out_w)
    oc = rem // (out_d * out_h * out_w)
    rem = rem % (out_d * out_h * out_w)
    od = rem // (out_h * out_w)
    rem = rem % (out_h * out_w)
    oh = rem // out_w
    ow = rem % out_w

    acc = 0.0
    for ic in range(in_c):
        for kd in range(k_d):
            for kh in range(k_h):
                for kw in range(k_w):
                    id_ = od * stride - pad + kd
                    ih = oh * stride - pad + kh
                    iw = ow * stride - pad + kw
                    if id_ >= 0 and id_ < in_d and ih >= 0 and ih < in_h and iw >= 0 and iw < in_w:
                        w_val = gWeight[oc, ic, kd, kh, kw]
                        in_val = gInput[n, ic, id_, ih, iw]
                        acc += in_val * w_val
    acc += gBias[oc, 0, 0, 0, 0]
    add_val = gAdd[n, oc, od, oh, ow]
    acc += add_val

    # HardSwish: x * relu6(x+3)/6
    x = acc
    relu6 = min(max(x + 3.0, 0.0), 6.0)
    out_val = x * (relu6 / 6.0)
    gOutput[n, oc, od, oh, ow] = out_val

@cute.jit
def conv_transpose_add_hardswish_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mAdd: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, out_c: int, out_d: int, out_h: int, out_w: int,
    in_c: int, in_d: int, in_h: int, in_w: int,
    k_d: int, k_h: int, k_w: int, stride: int, pad: int, out_pad: int
):
    total_threads = batch_size * out_c * out_d * out_h * out_w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_threads, threads_per_block)
    conv_transpose_add_hardswish_kernel(
        mInput, mWeight, mBias, mAdd, mOutput,
        batch_size, out_c, out_d, out_h, out_w,
        in_c, in_d, in_h, in_w,
        k_d, k_h, k_w, stride, pad, out_pad
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = None

    def forward(self, x, add_input):
        batch_size = x.size(0)
        in_d, in_h, in_w = x.size(2), x.size(3), x.size(4)
        out_d = (in_d - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        x = x.contiguous().cuda()
        add_input = add_input.contiguous().cuda()
        output = torch.empty(batch_size, self.out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))
        mWeight = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))
        mAdd = from_dlpack(add_input, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))

        if self.compiled is None:
            self.compiled = cute.compile(
                conv_transpose_add_hardswish_host,
                mInput, mWeight, mBias, mAdd, mOutput,
                batch_size, self.out_channels, out_d, out_h, out_w,
                self.in_channels, in_d, in_h, in_w,
                self.kernel_size, self.kernel_size, self.kernel_size, self.stride, self.padding, self.output_padding
            )

        self.compiled(
            mInput, mWeight, mBias, mAdd, mOutput,
            batch_size, self.out_channels, out_d, out_h, out_w,
            self.in_channels, in_d, in_h, in_w,
            self.kernel_size, self.kernel_size, self.kernel_size, self.stride, self.padding, self.output_padding
        )
        return output