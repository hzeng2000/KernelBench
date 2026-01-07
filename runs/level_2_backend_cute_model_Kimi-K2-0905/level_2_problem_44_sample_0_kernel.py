import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_transpose_scale_kernel(
    gInput: cute.Tensor,
    gWeight: cute.Tensor,
    gBias: cute.Tensor,
    gOutput: cute.Tensor,
    scale: float,
    batch: int, in_h: int, in_w: int,
    out_h: int, out_w: int,
    in_c: int, out_c: int,
    k_h: int, k_w: int,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
    out_pad_h: int, out_pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_n = bidz
    out_c_idx = bidy * bdimy + tidy
    out_h_idx = bidx * bdimx + tidx

    if out_n < batch and out_c_idx < out_c and out_h_idx < out_h:
        acc = 0.0
        for in_c_idx in range(in_c):
            for k_h_idx in range(k_h):
                for k_w_idx in range(k_w):
                    in_h_idx = (out_h_idx + pad_h - k_h_idx * stride_h - out_pad_h) // stride_h
                    in_w_idx = (out_w_idx + pad_w - k_w_idx * stride_w - out_pad_w) // stride_w
                    if in_h_idx >= 0 and in_h_idx < in_h and in_w_idx >= 0 and in_w_idx < in_w:
                        inp_val = gInput[out_n, in_c_idx, in_h_idx, in_w_idx]
                        w_val = gWeight[out_c_idx, in_c_idx, k_h_idx, k_w_idx]
                        acc += inp_val * w_val
        gOutput[out_n, out_c_idx, out_h_idx, out_w_idx] = acc * scale

@cute.kernel
def global_avg_pool_kernel(
    gInput: cute.Tensor,
    gOutput: cute.Tensor,
    batch: int, channels: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    n = bidz * bdimz + tidz
    c = bidy * bdimy + tidy

    if n < batch and c < channels:
        sum_val = 0.0
        for h in range(height):
            for w in range(width):
                sum_val += gInput[n, c, h, w]
        avg_val = sum_val / (height * width)
        gOutput[n, c, 0, 0] = avg_val

@cute.jit
def conv_transpose_scale_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    scale: float,
    batch: int, in_h: int, in_w: int,
    out_h: int, out_w: int,
    in_c: int, out_c: int,
    k_h: int, k_w: int,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
    out_pad_h: int, out_pad_w: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(out_h, 16)
    grid_y = cute.ceil_div(out_c, 16)
    grid_z = batch
    conv_transpose_scale_kernel(
        mInput, mWeight, mBias, mOutput, scale,
        batch, in_h, in_w, out_h, out_w, in_c, out_c,
        k_h, k_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(16, 16, 4))

@cute.jit
def global_avg_pool_host(
    mInput: cute.Tensor, mOutput: cute.Tensor,
    batch: int, channels: int, height: int, width: int
):
    threads_per_block = 256
    grid_x = 1
    grid_y = cute.ceil_div(channels, 32)
    grid_z = cute.ceil_div(batch, 8)
    global_avg_pool_kernel(mInput, mOutput, batch, channels, height, width).launch(
        grid=(grid_x, grid_y, grid_z), block=(1, 32, 8)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.compiled = {}

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros_like(weight[:,0,0,0])

        batch, in_c, in_h, in_w = x.shape
        out_c = weight.shape[0]
        k_h, k_w = self.conv_transpose.kernel_size
        stride_h, stride_w = self.conv_transpose.stride
        pad_h, pad_w = self.conv_transpose.padding
        out_pad_h, out_pad_w = self.conv_transpose.output_padding

        out_h = (in_h - 1) * stride_h - 2 * pad_h + k_h + out_pad_h
        out_w = (in_w - 1) * stride_w - 2 * pad_w + k_w + out_pad_w

        x_out = torch.empty(batch, out_c, out_h, out_w, dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(x_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_transpose_scale_host, mInput, mWeight, mBias, mOutput, self.multiplier,
                                    batch, in_h, in_w, out_h, out_w, in_c, out_c,
                                    k_h, k_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w)
            self.compiled[key] = compiled

        compiled(mInput, mWeight, mBias, mOutput, self.multiplier,
                 batch, in_h, in_w, out_h, out_w, in_c, out_c,
                 k_h, k_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w)

        # First global average pooling
        x_pooled1 = torch.empty(batch, out_c, 1, 1, dtype=x.dtype, device=x.device)
        mPooled1 = from_dlpack(x_pooled1, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3))
        compiled_pool1 = cute.compile(global_avg_pool_host, mOutput, mPooled1, batch, out_c, out_h, out_w)
        compiled_pool1(mOutput, mPooled1, batch, out_c, out_h, out_w)

        # Second global average pooling (redundant but kept as per spec)
        x_pooled2 = torch.empty(batch, out_c, 1, 1, dtype=x.dtype, device=x.device)
        mPooled2 = from_dlpack(x_pooled2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3))
        compiled_pool2 = cute.compile(global_avg_pool_host, mPooled1, mPooled2, batch, out_c, 1, 1)
        compiled_pool2(mPooled1, mPooled2, batch, out_c, 1, 1)

        return x_pooled2