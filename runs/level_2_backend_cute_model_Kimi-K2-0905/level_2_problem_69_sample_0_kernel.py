import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_hardswish_relu_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    height: int, width: int, kernel_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    gdimx, gdimy, gdimz = cute.arch.grid_dim()

    out_h = height - kernel_size + 1
    out_w = width - kernel_size + 1

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_c = bidz * bdimz + tidz

    if out_x < out_w and out_y < out_h and out_c < out_channels:
        acc = 0.0
        for n in range(batch_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    for ic in range(in_channels):
                        in_y = out_y + kh
                        in_x = out_x + kw
                        x_val = gX[n, ic, in_y, in_x]
                        w_val = gW[out_c, ic, kh, kw]
                        acc += x_val * w_val
            acc += gB[out_c]
            
            # HardSwish
            hardswish = acc * torch.nn.functional.hardswish(acc)
            # ReLU
            relu = max(0.0, hardswish)
            
            gY[n, out_c, out_y, out_x] = relu

@cute.jit
def conv_hardswish_relu_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    height: int, width: int, kernel_size: int
):
    out_h = height - kernel_size + 1
    out_w = width - kernel_size + 1

    threads_per_block = 8
    grid_x = cute.ceil_div(out_w, threads_per_block)
    grid_y = cute.ceil_div(out_h, threads_per_block)
    grid_z = cute.ceil_div(out_channels, threads_per_block)

    conv_hardswish_relu_kernel(
        mX, mW, mB, mY,
        batch_size, in_channels, out_channels,
        height, width, kernel_size
    ).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(threads_per_block, threads_per_block, threads_per_block)
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.compiled = {}

    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Get weight and bias from conv layer
        weight = self.conv.weight
        bias = self.conv.bias
        
        # Allocate output tensor
        out_h = height - self.kernel_size + 1
        out_w = width - self.kernel_size + 1
        y = torch.empty(batch_size, self.out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        # Compile and launch kernel
        key = (x.dtype, batch_size, self.in_channels, self.out_channels, height, width, self.kernel_size)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_hardswish_relu_host,
                mX, mW, mB, mY,
                batch_size, self.in_channels, self.out_channels,
                height, width, self.kernel_size
            )
            self.compiled[key] = compiled
        
        compiled(mX, mW, mB, mY, batch_size, self.in_channels, self.out_channels, height, width, self.kernel_size)
        return y