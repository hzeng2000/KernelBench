import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose_softmax_bias_scale_sigmoid_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor,
    gOutput: cute.Tensor, scaling_factor: float,
    batch_size: int, in_channels: int, out_channels: int,
    in_h: int, in_w: int, out_h: int, out_w: int,
    kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
    pad_h: int, pad_w: int, out_pad_h: int, out_pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    global_tid = bidz * bdimx * bdimy * bdimz + bidy * bdimx * bdimy + bidx * bdimx + tid
    
    total_threads = batch_size * out_channels * out_h * out_w
    if global_tid >= total_threads:
        return
    
    n = global_tid // (out_channels * out_h * out_w)
    c = (global_tid // (out_h * out_w)) % out_channels
    h = (global_tid // out_w) % out_h
    w = global_tid % out_w
    
    sum_val = 0.0
    
    for ic in range(in_channels):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_h_idx = (h + pad_h - kh * stride_h) // stride_h
                in_w_idx = (w + pad_w - kw * stride_w) // stride_w
                
                if (h + pad_h - kh * stride_h) % stride_h == 0 and (w + pad_w - kw * stride_w) % stride_w == 0:
                    if in_h_idx >= 0 and in_h_idx < in_h and in_w_idx >= 0 and in_w_idx < in_w:
                        weight_val = gWeight[c, ic, kh, kw]
                        input_val = gInput[n, ic, in_h_idx, in_w_idx]
                        sum_val += input_val * weight_val
    
    gOutput[n, c, h, w] = sum_val

@cute.kernel
def softmax_bias_scale_sigmoid_kernel(
    gInput: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    scaling_factor: float, batch_size: int, channels: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    global_tid = bidz * bdimx * bdimy * bdimz + bidy * bdimx * bdimy + bidx * bdimx + tid
    
    total_threads = batch_size * height * width
    if global_tid >= total_threads:
        return
    
    n = global_tid // (height * width)
    h = (global_tid // width) % height
    w = global_tid % width
    
    max_val = -float('inf')
    for c in range(channels):
        val = gInput[n, c, h, w]
        if val > max_val:
            max_val = val
    
    sum_exp = 0.0
    for c in range(channels):
        val = gInput[n, c, h, w]
        exp_val = math.exp(val - max_val)
        sum_exp += exp_val
    
    for c in range(channels):
        val = gInput[n, c, h, w]
        exp_val = math.exp(val - max_val)
        softmax_val = exp_val / sum_exp
        bias_val = gBias[c, 0, 0]
        scaled_val = (softmax_val + bias_val) * scaling_factor
        sigmoid_val = 1.0 / (1.0 + math.exp(-scaled_val))
        gOutput[n, c, h, w] = sigmoid_val

@cute.jit
def fused_conv_transpose_softmax_bias_scale_sigmoid_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    scaling_factor: float, batch_size: int, in_channels: int, out_channels: int,
    in_h: int, in_w: int, out_h: int, out_w: int,
    kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
    pad_h: int, pad_w: int, out_pad_h: int, out_pad_w: int
):
    threads_per_block = 256
    total_conv_elems = batch_size * out_channels * out_h * out_w
    grid_x = cute.ceil_div(total_conv_elems, threads_per_block)
    
    temp_output = torch.empty((batch_size, out_channels, out_h, out_w), dtype=torch.float32, device=mInput.device)
    mTemp = from_dlpack(temp_output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
    
    conv_transpose_softmax_bias_scale_sigmoid_kernel(
        mInput, mWeight, mBias, mTemp, scaling_factor,
        batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))
    
    threads_per_block_softmax = 256
    total_softmax_elems = batch_size * out_h * out_w
    grid_x_softmax = cute.ceil_div(total_softmax_elems, threads_per_block_softmax)
    
    softmax_bias_scale_sigmoid_kernel(
        mTemp, mBias, mOutput, scaling_factor,
        batch_size, out_channels, out_h, out_w
    ).launch(grid=(grid_x_softmax, 1, 1), block=(threads_per_block_softmax, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.compiled = {}

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        x = x.contiguous().cuda()
        output = torch.empty((batch_size, self.out_channels, out_h, out_w), dtype=torch.float32, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_conv_transpose_softmax_bias_scale_sigmoid_host,
                mInput, mWeight, mBias, mOutput, self.scaling_factor,
                batch_size, self.in_channels, self.out_channels,
                in_h, in_w, out_h, out_w,
                self.kernel_size, self.kernel_size, self.stride, self.stride,
                self.padding, self.padding, self.output_padding, self.output_padding
            )
            self.compiled[key] = compiled
        
        compiled(
            mInput, mWeight, mBias, mOutput, self.scaling_factor,
            batch_size, self.in_channels, self.out_channels,
            in_h, in_w, out_h, out_w,
            self.kernel_size, self.kernel_size, self.stride, self.stride,
            self.padding, self.padding, self.output_padding, self.output_padding
        )
        
        return output