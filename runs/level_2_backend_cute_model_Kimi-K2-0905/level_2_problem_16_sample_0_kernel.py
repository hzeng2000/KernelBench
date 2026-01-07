import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_transpose_mish_add_hardtanh_scale_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int, in_h: int, in_w: int,
    out_h: int, out_w: int, kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
    pad_h: int, pad_w: int, out_pad_h: int, out_pad_w: int, add_val: float, scale_val: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_id = tidz * bdimx * bdimy + tidy * bdimx + tidx
    block_id = bidz * cute.arch.grid_dim_x() * cute.arch.grid_dim_y() + bidy * cute.arch.grid_dim_x() + bidx
    threads_per_block = bdimx * bdimy * bdimz

    global_thread_id = block_id * threads_per_block + thread_id

    total_output_pixels = batch_size * out_channels * out_h * out_w
    if global_thread_id >= total_output_pixels:
        return

    oc = global_thread_id % out_channels
    tmp = global_thread_id // out_channels
    ow = tmp % out_w
    tmp = tmp // out_w
    oh = tmp % out_h
    b = tmp // out_h

    out_y = oh
    out_x = ow

    acc = 0.0
    for ic in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_y = (out_y + pad_h - kh * 1) // stride_h
                in_x = (out_x + pad_w - kw * 1) // stride_w
                if (out_y + pad_h - kh) % stride_h != 0 or (out_x + pad_w - kw) % stride_w != 0:
                    continue
                if in_y < 0 or in_x < 0 or in_y >= in_h or in_x >= in_w:
                    continue
                in_val = gInput[b, ic, in_y, in_x]
                w_val = gWeight[oc, ic, kh, kw]
                acc += in_val * w_val

    if gBias.shape[0] > 0:
        acc += gBias[oc]

    # Mish activation
    x = acc
    softplus = cute.math.logf(1.0 + cute.math.expf(x))
    mish_val = x * cute.math.tanh(softplus)

    # Add value
    mish_val += add_val

    # Hardtanh
    hardtanh_val = mish_val
    if hardtanh_val < -1.0:
        hardtanh_val = -1.0
    elif hardtanh_val > 1.0:
        hardtanh_val = 1.0

    # Scale
    final_val = hardtanh_val * scale_val

    gOutput[b, oc, out_y, out_x] = final_val

@cute.jit
def conv_transpose_mish_add_hardtanh_scale_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int, in_h: int, in_w: int,
    out_h: int, out_w: int, kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
    pad_h: int, pad_w: int, out_pad_h: int, out_pad_w: int, add_val: float, scale_val: float
):
    threads_per_block = 256
    total_output_pixels = batch_size * out_channels * out_h * out_w
    grid_x = cute.ceil_div(total_output_pixels, threads_per_block)

    conv_transpose_mish_add_hardtanh_scale_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_channels, out_channels, in_h, in_w,
        out_h, out_w, kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, out_pad_h, out_pad_w, add_val, scale_val
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.compiled = {}

    def forward(self, x):
        batch_size, in_c, in_h, in_w = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        out_pad_h, out_pad_w = self.output_padding

        out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
        out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w

        x = x.contiguous().cuda()
        output = torch.empty(batch_size, self.out_channels, out_h, out_w, dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_transpose_mish_add_hardtanh_scale_host,
                mInput, mWeight, mBias, mOutput,
                batch_size, in_c, self.out_channels, in_h, in_w,
                out_h, out_w, kernel_h, kernel_w, stride_h, stride_w,
                pad_h, pad_w, out_pad_h, out_pad_w, self.add_value, self.scale
            )
            self.compiled[key] = compiled

        compiled(
            mInput, mWeight, mBias, mOutput,
            batch_size, in_c, self.out_channels, in_h, in_w,
            out_h, out_w, kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, out_pad_h, out_pad_w, self.add_value, self.scale
        )
        return output