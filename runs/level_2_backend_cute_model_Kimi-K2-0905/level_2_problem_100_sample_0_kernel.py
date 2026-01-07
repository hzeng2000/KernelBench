import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose3d_clamp_div_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    min_val: float, divisor: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_idx = (bidz * bdimz + tidz) * bdimy * bdimx + (bidy * bdimy + tidy) * bdimx + (bidx * bdimx + tidx)

    total_threads = batch_size * out_channels * out_d * out_h * out_w
    if thread_idx >= total_threads:
        return

    # Compute output position
    tmp = thread_idx
    ow = tmp % out_w
    tmp = tmp // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    tmp = tmp // out_d
    oc = tmp % out_channels
    tmp = tmp // out_channels
    b = tmp % batch_size

    # Compute input region
    acc = 0.0
    for ic in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    in_d_idx = od * stride_d - pad_d + kd
                    in_h_idx = oh * stride_h - pad_h + kh
                    in_w_idx = ow * stride_w - pad_w + kw
                    if in_d_idx >= 0 and in_d_idx < in_d and in_h_idx >= 0 and in_h_idx < in_h and in_w_idx >= 0 and in_w_idx < in_w:
                        acc += gInput[b, ic, in_d_idx, in_h_idx, in_w_idx] * gWeight[oc, ic, kd, kh, kw]

    if gBias.shape[0] > 0:
        acc += gBias[oc]

    # Clamp and divide
    acc = max(acc, min_val)
    acc = acc / divisor

    gOutput[b, oc, od, oh, ow] = acc

@cute.jit
def conv_transpose3d_clamp_div_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    min_val: float, divisor: float
):
    total_elems = batch_size * out_channels * out_d * out_h * out_w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    conv_transpose3d_clamp_div_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        min_val, divisor
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get conv weights and bias
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        # Input shape
        batch_size, in_channels, in_d, in_h, in_w = x.shape
        out_channels = weight.shape[0]
        kernel_d, kernel_h, kernel_w = weight.shape[2:]
        stride_d, stride_h, stride_w = self.conv_transpose.stride
        pad_d, pad_h, pad_w = self.conv_transpose.padding

        # Compute output size
        out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d
        out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h
        out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w

        # Allocate output
        output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

        # Make tensors contiguous and aligned
        x = x.contiguous().cuda()
        weight = weight.contiguous().cuda()
        bias = bias.contiguous().cuda()

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_transpose3d_clamp_div_host,
                mInput, mWeight, mBias, mOutput,
                batch_size, in_channels, out_channels,
                in_d, in_h, in_w,
                out_d, out_h, out_w,
                kernel_d, kernel_h, kernel_w,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                self.min_value, self.divisor
            )
            self.compiled[key] = compiled

        compiled(
            mInput, mWeight, mBias, mOutput,
            batch_size, in_channels, out_channels,
            in_d, in_h, in_w,
            out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            self.min_value, self.divisor
        )
        return output