import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_transpose_sub_tanh_kernel(
    gI: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gO: cute.Tensor,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
    out_pad_h: int, out_pad_w: int,
    batch: int, in_c: int, in_h: int, in_w: int,
    out_c: int, out_h: int, out_w: int,
    kernel_h: int, kernel_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidz * bdimz + tidz
    out_y = bidy * bdimy + tidy
    out_n = bidx * bdimx + tidx

    if out_n < batch and out_y < out_h and out_x < out_w:
        acc = 0.0
        for kc in range(in_c):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    in_y = (out_y + pad_h - kh * stride_h) // stride_h
                    in_x = (out_x + pad_w - kw * stride_w) // stride_w
                    if in_y >= 0 and in_y < in_h and in_x >= 0 and in_x < in_w:
                        w_val = gW[kc, out_c, kernel_h - 1 - kh, kernel_w - 1 - kw]
                        i_val = gI[out_n, kc, in_y, in_x]
                        acc += w_val * i_val
        acc -= gB[out_c, 0, 0]
        gO[out_n, out_c, out_y, out_x] = cute.math.tanh(acc)

@cute.jit
def conv_transpose_sub_tanh_host(
    mI: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mO: cute.Tensor,
    stride_h: int, stride_w: int,
    pad_h: int, pad_w: int,
    out_pad_h: int, out_pad_w: int
):
    batch, in_c, in_h, in_w = mI.shape
    out_c, _, kernel_h, kernel_w = mW.shape
    out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
    out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w

    threads_per_block = 8
    grid_x = cute.ceil_div(batch, threads_per_block)
    grid_y = cute.ceil_div(out_h, threads_per_block)
    grid_z = cute.ceil_div(out_w, threads_per_block)

    for oc in range(out_c):
        conv_transpose_sub_tanh_kernel(
            mI, mW, mB, mO,
            stride_h, stride_w,
            pad_h, pad_w,
            out_pad_h, out_pad_w,
            batch, in_c, in_h, in_w,
            oc, out_h, out_w,
            kernel_h, kernel_w
        ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self.compiled = {}

    def forward(self, x):
        batch, in_c, in_h, in_w = x.shape
        out_h = (in_h - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        out_w = (in_w - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        output = torch.empty(batch, self.out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        mI = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mO = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_transpose_sub_tanh_host, mI, mW, mB, mO,
                                    self.stride[0], self.stride[1],
                                    self.padding[0], self.padding[1],
                                    self.output_padding[0], self.output_padding[1])
            self.compiled[key] = compiled
        
        compiled(mI, mW, mB, mO,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.output_padding[0], self.output_padding[1])
        
        return output