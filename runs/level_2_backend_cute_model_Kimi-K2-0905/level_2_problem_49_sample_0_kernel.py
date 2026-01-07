import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose3d_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz
    
    if out_x < out_w and out_y < out_h and out_z < out_d:
        for n in range(batch_size):
            for oc in range(out_channels):
                acc = 0.0
                for ic in range(in_channels):
                    for kd in range(kernel_d):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                in_z = out_z + pad_d - kd * stride_d - out_pad_d
                                in_y = out_y + pad_h - kh * stride_h - out_pad_h
                                in_x = out_x + pad_w - kw * stride_w - out_pad_w
                                
                                if in_z >= 0 and in_z < in_d and in_y >= 0 and in_y < in_h and in_x >= 0 and in_x < in_w:
                                    acc += gInput[n, ic, in_z, in_y, in_x] * gWeight[oc, ic, kd, kh, kw]
                
                if gBias.shape[0] > 0:
                    acc += gBias[oc]
                gOutput[n, oc, out_z, out_y, out_x] = acc

@cute.kernel
def softmax_sigmoid_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    w = bidx * bdimx + tidx
    h = bidy * bdimy + tidy
    d = bidz * bdimz + tidz
    
    if w < width and h < height and d < depth:
        for n in range(batch_size):
            max_val = -float('inf')
            for c in range(channels):
                val = gInput[n, c, d, h, w]
                if val > max_val:
                    max_val = val
            
            sum_exp = 0.0
            for c in range(channels):
                exp_val = math.exp(gInput[n, c, d, h, w] - max_val)
                sum_exp += exp_val
                gOutput[n, c, d, h, w] = exp_val
            
            for c in range(channels):
                softmax_val = gOutput[n, c, d, h, w] / sum_exp
                sigmoid_val = 1.0 / (1.0 + math.exp(-softmax_val))
                gOutput[n, c, d, h, w] = sigmoid_val

@cute.jit
def conv_transpose3d_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int
):
    threads_per_block = 256
    blocks_x = cute.ceil_div(out_w, 8)
    blocks_y = cute.ceil_div(out_h, 8)
    blocks_z = cute.ceil_div(out_d, 8)
    
    conv_transpose3d_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w
    ).launch(grid=(blocks_x, blocks_y, blocks_z), block=(8, 8, 8))

@cute.jit
def softmax_sigmoid_host(
    mInput: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int
):
    threads_per_block = 256
    blocks_x = cute.ceil_div(width, 8)
    blocks_y = cute.ceil_div(height, 8)
    blocks_z = cute.ceil_div(depth, 8)
    
    softmax_sigmoid_kernel(
        mInput, mOutput,
        batch_size, channels, depth, height, width
    ).launch(grid=(blocks_x, blocks_y, blocks_z), block=(8, 8, 8))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        
        out_d = (in_d - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0] + self.output_padding[0]
        out_h = (in_h - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1] + self.output_padding[1]
        out_w = (in_w - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2] + self.output_padding[2]
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda() if self.bias is not None else torch.empty(0).cuda()
        
        conv_output = torch.empty(batch_size, self.out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mConvOutput = from_dlpack(conv_output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled_conv = cute.compile(conv_transpose3d_host, mInput, mWeight, mBias, mConvOutput,
                                       batch_size, self.in_channels, self.out_channels,
                                       in_d, in_h, in_w, out_d, out_h, out_w,
                                       self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                                       self.stride[0], self.stride[1], self.stride[2],
                                       self.padding[0], self.padding[1], self.padding[2],
                                       self.output_padding[0], self.output_padding[1], self.output_padding[2])
            compiled_softmax = cute.compile(softmax_sigmoid_host, mConvOutput, mConvOutput,
                                          batch_size, self.out_channels, out_d, out_h, out_w)
            self.compiled[key] = (compiled_conv, compiled_softmax)
        
        compiled_conv, compiled_softmax = self.compiled[key]
        compiled_conv(mInput, mWeight, mBias, mConvOutput,
                     batch_size, self.in_channels, self.out_channels,
                     in_d, in_h, in_w, out_d, out_h, out_w,
                     self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                     self.stride[0], self.stride[1], self.stride[2],
                     self.padding[0], self.padding[1], self.padding[2],
                     self.output_padding[0], self.output_padding[1], self.output_padding[2])
        
        compiled_softmax(mConvOutput, mConvOutput,
                        batch_size, self.out_channels, out_d, out_h, out_w)
        
        return conv_output