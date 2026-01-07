import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv2d_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    height: int, width: int, kernel_size: int, out_height: int, out_width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_c = bidz * bdimz + tidz
    out_h = bidy * bdimy + tidy
    out_w = bidx * bdimx + tidx
    
    if out_c < out_channels and out_h < out_height and out_w < out_width:
        sum_val = 0.0
        for b in range(batch_size):
            for ic in range(in_channels):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        in_h = out_h + kh
                        in_w = out_w + kw
                        if in_h < height and in_w < width:
                            sum_val += gX[b, ic, in_h, in_w] * gW[out_c, ic, kh, kw]
            gY[b, out_c, out_h, out_w] = sum_val + gB[out_c]

@cute.kernel
def avgpool_sigmoid_kernel(
    gX: cute.Tensor, gY: cute.Tensor, batch_size: int, channels: int,
    in_height: int, in_width: int, pool_size: int, out_height: int, out_width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    b = bidz * bdimz + tidz
    c = bidy * bdimy + tidy
    out_h = bidx * bdimx + tidx
    
    if b < batch_size and c < channels and out_h < out_height:
        for out_w in range(out_width):
            sum_val = 0.0
            count = 0
            for ph in range(pool_size):
                for pw in range(pool_size):
                    in_h = out_h * pool_size + ph
                    in_w = out_w * pool_size + pw
                    if in_h < in_height and in_w < in_width:
                        sum_val += gX[b, c, in_h, in_w]
                        count += 1
            avg_val = sum_val / count if count > 0 else 0.0
            sigmoid_val = 1.0 / (1.0 + math.exp(-avg_val))
            gY[b, c, out_h, out_w] = sigmoid_val

@cute.kernel
def sum_reduce_kernel(
    gX: cute.Tensor, gY: cute.Tensor,
    batch_size: int, channels: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    b = bidz * bdimz + tidz
    
    if b < batch_size:
        sum_val = 0.0
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    sum_val += gX[b, c, h, w]
        gY[b] = sum_val

@cute.jit
def conv2d_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    height: int, width: int, kernel_size: int, out_height: int, out_width: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(out_width, threads_per_block)
    grid_y = cute.ceil_div(out_height, threads_per_block)
    grid_z = cute.ceil_div(out_channels, threads_per_block)
    
    conv2d_kernel(mX, mW, mB, mY, batch_size, in_channels, out_channels,
                  height, width, kernel_size, out_height, out_width).launch(
        grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block)
    )

@cute.jit
def avgpool_sigmoid_host(
    mX: cute.Tensor, mY: cute.Tensor, batch_size: int, channels: int,
    in_height: int, in_width: int, pool_size: int, out_height: int, out_width: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(out_width, threads_per_block)
    grid_y = cute.ceil_div(out_height, threads_per_block)
    grid_z = cute.ceil_div(batch_size, threads_per_block)
    
    avgpool_sigmoid_kernel(mX, mY, batch_size, channels, in_height, in_width,
                           pool_size, out_height, out_width).launch(
        grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block)
    )

@cute.jit
def sum_reduce_host(
    mX: cute.Tensor, mY: cute.Tensor,
    batch_size: int, channels: int, height: int, width: int
):
    threads_per_block = 1
    grid_z = batch_size
    
    sum_reduce_kernel(mX, mY, batch_size, channels, height, width).launch(
        grid=(1, 1, grid_z), block=(threads_per_block, threads_per_block, threads_per_block)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.compiled = {}

    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1
        
        pool_out_height = out_height // self.pool_kernel_size
        pool_out_width = out_width // self.pool_kernel_size
        
        x = x.contiguous().cuda()
        conv_out = torch.empty(batch_size, self.out_channels, out_height, out_width, dtype=x.dtype, device=x.device)
        pool_out = torch.empty(batch_size, self.out_channels, pool_out_height, pool_out_width, dtype=x.dtype, device=x.device)
        final_out = torch.empty(batch_size, dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mConv = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mPool = from_dlpack(pool_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mFinal = from_dlpack(final_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = {
                'conv': cute.compile(conv2d_host, mX, mW, mB, mConv, batch_size, self.in_channels, self.out_channels,
                                     height, width, self.kernel_size, out_height, out_width),
                'pool': cute.compile(avgpool_sigmoid_host, mConv, mPool, batch_size, self.out_channels,
                                     out_height, out_width, self.pool_kernel_size, pool_out_height, pool_out_width),
                'sum': cute.compile(sum_reduce_host, mPool, mFinal, batch_size, self.out_channels,
                                    pool_out_height, pool_out_width)
            }
            self.compiled[key] = compiled
        
        compiled['conv'](mX, mW, mB, mConv, batch_size, self.in_channels, self.out_channels,
                         height, width, self.kernel_size, out_height, out_width)
        compiled['pool'](mConv, mPool, batch_size, self.out_channels,
                         out_height, out_width, self.pool_kernel_size, pool_out_height, pool_out_width)
        compiled['sum'](mPool, mFinal, batch_size, self.out_channels, pool_out_height, pool_out_width)
        
        return final_out