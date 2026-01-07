import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv3d_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    depth: int, height: int, width: int,
    kernel_size: int, stride: int, padding: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_d = bidx * bdimx + tidx
    out_h = bidy * bdimy + tidy
    out_w = bidz * bdimz + tidz

    if out_d < depth and out_h < height and out_w < width:
        for b in range(batch_size):
            for oc in range(out_channels):
                acc = 0.0
                for ic in range(in_channels):
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                in_d = out_d + kd - padding
                                in_h = out_h + kh - padding
                                in_w = out_w + kw - padding
                                if 0 <= in_d < depth and 0 <= in_h < height and 0 <= in_w < width:
                                    acc += gInput[b, ic, in_d, in_h, in_w] * gWeight[oc, ic, kd, kh, kw]
                gOutput[b, oc, out_d, out_h, out_w] = acc

@cute.kernel
def softmax_kernel(gInput: cute.Tensor, gOutput: cute.Tensor, batch_size: int, channels: int, depth: int, height: int, width: int):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    b = bidx * bdimx + tidx
    d = bidy * bdimy + tidy
    h = bidz * bdimz + tidz

    if b < batch_size and d < depth and h < height:
        for w in range(width):
            max_val = gInput[b, 0, d, h, w]
            for c in range(1, channels):
                max_val = max(max_val, gInput[b, c, d, h, w])
            
            sum_exp = 0.0
            for c in range(channels):
                sum_exp += cute.exp(gInput[b, c, d, h, w] - max_val)
            
            for c in range(channels):
                gOutput[b, c, d, h, w] = cute.exp(gInput[b, c, d, h, w] - max_val) / sum_exp

@cute.kernel
def maxpool3d_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, channels: int, in_depth: int, in_height: int, in_width: int,
    kernel_size: int, stride: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    b = bidx * bdimx + tidx
    d = bidy * bdimy + tidy
    h = bidz * bdimz + tidz

    out_depth = in_depth // stride
    out_height = in_height // stride
    out_width = in_width // stride

    if b < batch_size and d < out_depth and h < out_height:
        for w in range(out_width):
            max_val = -1e38
            for kd in range(kernel_size):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        in_d = d * stride + kd
                        in_h = h * stride + kh
                        in_w = w * stride + kw
                        if in_d < in_depth and in_h < in_height and in_w < in_width:
                            for c in range(channels):
                                val = gInput[b, c, in_d, in_h, in_w]
                                max_val = max(max_val, val)
            for c in range(channels):
                gOutput[b, c, d, h, w] = max_val

@cute.jit
def conv3d_host(gInput, gWeight, gOutput, batch_size, in_channels, out_channels, depth, height, width, kernel_size):
    threads_per_block = 8
    grid_x = cute.ceil_div(depth, threads_per_block)
    grid_y = cute.ceil_div(height, threads_per_block)
    grid_z = cute.ceil_div(width, threads_per_block)
    conv3d_kernel(gInput, gWeight, gOutput, batch_size, in_channels, out_channels, depth, height, width, kernel_size, 1, 0).launch(
        grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

@cute.jit
def softmax_host(gInput, gOutput, batch_size, channels, depth, height, width):
    threads_per_block = 8
    grid_x = cute.ceil_div(batch_size, threads_per_block)
    grid_y = cute.ceil_div(depth, threads_per_block)
    grid_z = cute.ceil_div(height, threads_per_block)
    softmax_kernel(gInput, gOutput, batch_size, channels, depth, height, width).launch(
        grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

@cute.jit
def maxpool3d_host(gInput, gOutput, batch_size, channels, in_depth, in_height, in_width, kernel_size, stride):
    threads_per_block = 8
    out_depth = in_depth // stride
    out_height = in_height // stride
    grid_x = cute.ceil_div(batch_size, threads_per_block)
    grid_y = cute.ceil_div(out_depth, threads_per_block)
    grid_z = cute.ceil_div(out_height, threads_per_block)
    maxpool3d_kernel(gInput, gOutput, batch_size, channels, in_depth, in_height, in_width, kernel_size, stride).launch(
        grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.compiled = {}

    def forward(self, x):
        batch_size, _, depth, height, width = x.shape
        x = x.contiguous().cuda()
        conv_out = torch.empty(batch_size, self.out_channels, depth, height, width, dtype=x.dtype, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(self.conv_weight.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = {
                'conv': cute.compile(conv3d_host, mInput, mWeight, mConvOut, batch_size, self.in_channels, self.out_channels, depth, height, width, self.kernel_size),
                'softmax': cute.compile(softmax_host, mConvOut, mConvOut, batch_size, self.out_channels, depth, height, width),
                'pool1': cute.compile(maxpool3d_host, mConvOut, mConvOut, batch_size, self.out_channels, depth, height, width, self.pool_kernel_size, self.pool_kernel_size),
                'pool2': cute.compile(maxpool3d_host, mConvOut, mConvOut, batch_size, self.out_channels, depth//2, height//2, width//2, self.pool_kernel_size, self.pool_kernel_size)
            }
            self.compiled[key] = compiled

        compiled['conv'](mInput, mWeight, mConvOut, batch_size, self.in_channels, self.out_channels, depth, height, width, self.kernel_size)
        compiled['softmax'](mConvOut, mConvOut, batch_size, self.out_channels, depth, height, width)
        
        pool1_out = torch.empty(batch_size, self.out_channels, depth//2, height//2, width//2, dtype=x.dtype, device=x.device)
        mPool1Out = from_dlpack(pool1_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        compiled['pool1'](mConvOut, mPool1Out, batch_size, self.out_channels, depth, height, width, self.pool_kernel_size, self.pool_kernel_size)
        
        pool2_out = torch.empty(batch_size, self.out_channels, depth//4, height//4, width//4, dtype=x.dtype, device=x.device)
        mPool2Out = from_dlpack(pool2_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        compiled['pool2'](mPool1Out, mPool2Out, batch_size, self.out_channels, depth//2, height//2, width//2, self.pool_kernel_size, self.pool_kernel_size)
        
        return pool2_out