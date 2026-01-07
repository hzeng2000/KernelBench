import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv3d_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_depth: int, in_height: int, in_width: int,
    out_depth: int, out_height: int, out_width: int,
    kernel_size: int, stride: int, padding: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_x < out_width and out_y < out_height and out_z < out_depth:
        for n in range(batch_size):
            for oc in range(out_channels):
                acc = 0.0
                if gBias is not None:
                    acc = gBias[oc]
                
                for ic in range(in_channels):
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                in_d = out_z * stride - padding + kd
                                in_h = out_y * stride - padding + kh
                                in_w = out_x * stride - padding + kw
                                
                                if in_d >= 0 and in_d < in_depth and in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                                    weight_val = gWeight[oc, ic, kd, kh, kw]
                                    input_val = gInput[n, ic, in_d, in_h, in_w]
                                    acc += weight_val * input_val
                
                gOutput[n, oc, out_z, out_y, out_x] = acc

@cute.kernel
def maxpool3d_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, channels: int, in_depth: int, in_height: int, in_width: int,
    out_depth: int, out_height: int, out_width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_x < out_width and out_y < out_height and out_z < out_depth:
        for n in range(batch_size):
            for c in range(channels):
                max_val = float('-inf')
                for kd in range(2):
                    for kh in range(2):
                        for kw in range(2):
                            in_d = out_z * 2 + kd
                            in_h = out_y * 2 + kh
                            in_w = out_x * 2 + kw
                            if in_d < in_depth and in_h < in_height and in_w < in_width:
                                val = gInput[n, c, in_d, in_h, in_w]
                                max_val = cute.max(max_val, val)
                gOutput[n, c, out_z, out_y, out_x] = max_val

@cute.kernel
def logsumexp_relu_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    x = bidx * bdimx + tidx
    y = bidy * bdimy + tidy
    z = bidz * bdimz + tidz

    if x < width and y < height and z < depth:
        for n in range(batch_size):
            max_val = float('-inf')
            for c in range(channels):
                val = gInput[n, c, z, y, x]
                if val > max_val:
                    max_val = val
            
            sum_exp = 0.0
            for c in range(channels):
                val = gInput[n, c, z, y, x]
                sum_exp += cute.exp(val - max_val)
            
            log_sum = max_val + cute.log(sum_exp)
            relu_val = cute.max(0.0, log_sum)
            gOutput[n, 0, z, y, x] = relu_val

@cute.jit
def conv3d_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_depth: int, in_height: int, in_width: int,
    out_depth: int, out_height: int, out_width: int,
    kernel_size: int, stride: int, padding: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(out_width, threads_per_block)
    grid_y = cute.ceil_div(out_height, threads_per_block)
    grid_z = cute.ceil_div(out_depth, threads_per_block)
    
    conv3d_kernel(mInput, mWeight, mBias, mOutput,
                  batch_size, in_channels, out_channels,
                  in_depth, in_height, in_width,
                  out_depth, out_height, out_width,
                  kernel_size, stride, padding).launch(
                      grid=(grid_x, grid_y, grid_z),
                      block=(threads_per_block, threads_per_block, threads_per_block))

@cute.jit
def maxpool3d_host(
    mInput: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, channels: int, in_depth: int, in_height: int, in_width: int,
    out_depth: int, out_height: int, out_width: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(out_width, threads_per_block)
    grid_y = cute.ceil_div(out_height, threads_per_block)
    grid_z = cute.ceil_div(out_depth, threads_per_block)
    
    maxpool3d_kernel(mInput, mOutput,
                     batch_size, channels, in_depth, in_height, in_width,
                     out_depth, out_height, out_width).launch(
                         grid=(grid_x, grid_y, grid_z),
                         block=(threads_per_block, threads_per_block, threads_per_block))

@cute.jit
def logsumexp_relu_host(
    mInput: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(width, threads_per_block)
    grid_y = cute.ceil_div(height, threads_per_block)
    grid_z = cute.ceil_div(depth, threads_per_block)
    
    logsumexp_relu_kernel(mInput, mOutput,
                          batch_size, channels, depth, height, width).launch(
                              grid=(grid_x, grid_y, grid_z),
                              block=(threads_per_block, threads_per_block, threads_per_block))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_depth = x.shape[2]
        in_height = x.shape[3]
        in_width = x.shape[4]
        
        out_depth = (in_depth + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        x = x.contiguous().cuda()
        conv_out = torch.empty(batch_size, self.conv_weight.shape[0], out_depth, out_height, out_width, dtype=x.dtype, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(self.conv_weight.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(self.conv_bias.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv3d_host, mInput, mWeight, mBias, mConvOut,
                                    batch_size, in_channels, self.conv_weight.shape[0],
                                    in_depth, in_height, in_width,
                                    out_depth, out_height, out_width,
                                    self.kernel_size, self.stride, self.padding)
            self.compiled[key] = compiled
        
        compiled(mInput, mWeight, mBias, mConvOut,
                 batch_size, in_channels, self.conv_weight.shape[0],
                 in_depth, in_height, in_width,
                 out_depth, out_height, out_width,
                 self.kernel_size, self.stride, self.padding)
        
        pool_out_depth = out_depth // 2
        pool_out_height = out_height // 2
        pool_out_width = out_width // 2
        pool_out = torch.empty(batch_size, self.conv_weight.shape[0], pool_out_depth, pool_out_height, pool_out_width, dtype=x.dtype, device=x.device)
        
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mPoolOut = from_dlpack(pool_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        compiled_pool = cute.compile(maxpool3d_host, mConvOut, mPoolOut,
                                     batch_size, self.conv_weight.shape[0], out_depth, out_height, out_width,
                                     pool_out_depth, pool_out_height, pool_out_width)
        compiled_pool(mConvOut, mPoolOut,
                      batch_size, self.conv_weight.shape[0], out_depth, out_height, out_width,
                      pool_out_depth, pool_out_height, pool_out_width)
        
        final_out = torch.empty(batch_size, 1, pool_out_depth, pool_out_height, pool_out_width, dtype=x.dtype, device=x.device)
        mPoolOut = from_dlpack(pool_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mFinalOut = from_dlpack(final_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        compiled_final = cute.compile(logsumexp_relu_host, mPoolOut, mFinalOut,
                                      batch_size, self.conv_weight.shape[0], pool_out_depth, pool_out_height, pool_out_width)
        compiled_final(mPoolOut, mFinalOut,
                       batch_size, self.conv_weight.shape[0], pool_out_depth, pool_out_height, pool_out_width)
        
        return final_out