import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_transpose_conv_mean_bias_softmax_tanh_scale_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int, depth: int, height: int, width: int,
    kernel_d: int, kernel_h: int, kernel_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int, scaling_factor: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    total_threads = bdimx * bdimy * bdimz
    
    out_depth = (depth + 2 * pad_d - kernel_d) // stride_d + 1
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    total_elements = batch_size * out_channels * out_height * out_width
    
    for idx in range(tid, total_elements, total_threads):
        b = idx // (out_channels * out_height * out_width)
        c = (idx // (out_height * out_width)) % out_channels
        h = (idx // out_width) % out_height
        w = idx % out_width
        
        sum_val = 0.0
        for ic in range(in_channels):
            for kd in range(kernel_d):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        in_d = (b * stride_d + kd - pad_d)
                        in_h = (h * stride_h + kh - pad_h)
                        in_w = (w * stride_w + kw - pad_w)
                        
                        if in_d >= 0 and in_d < depth and in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                            input_val = gInput[b, ic, in_d, in_h, in_w]
                            weight_val = gWeight[c, ic, kd, kh, kw]
                            sum_val += input_val * weight_val
        
        mean_val = sum_val / depth
        bias_val = gBias[0, c, 0, 0, 0]
        biased_val = mean_val + bias_val
        
        gOutput[b, c, 0, h, w] = biased_val

@cute.kernel
def softmax_channel_kernel(gInput: cute.Tensor, gOutput: cute.Tensor, batch_size: int, channels: int, spatial_size: int):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    total_threads = bdimx * bdimy * bdimz
    
    for idx in range(tid, batch_size * spatial_size, total_threads):
        b = idx // spatial_size
        hw = idx % spatial_size
        
        max_val = -float('inf')
        for c in range(channels):
            val = gInput[b, c, 0, hw // gInput.shape[4], hw % gInput.shape[4]]
            if val > max_val:
                max_val = val
        
        sum_exp = 0.0
        for c in range(channels):
            val = gInput[b, c, 0, hw // gInput.shape[4], hw % gInput.shape[4]]
            exp_val = math.exp(val - max_val)
            gOutput[b, c, 0, hw // gInput.shape[4], hw % gInput.shape[4]] = exp_val
            sum_exp += exp_val
        
        for c in range(channels):
            gOutput[b, c, 0, hw // gInput.shape[4], hw % gInput.shape[4]] /= sum_exp

@cute.kernel
def tanh_scale_kernel(gInput: cute.Tensor, gOutput: cute.Tensor, scaling_factor: float, total_elements: int):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    
    for idx in range(tid, total_elements, bdimx * bdimy * bdimz):
        val = gInput[idx]
        tanh_val = math.tanh(val)
        gOutput[idx] = tanh_val * scaling_factor

@cute.jit
def fused_transpose_conv_mean_bias_kernel_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mTemp: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int, depth: int, height: int, width: int,
    kernel_size: int, stride: int, padding: int, scaling_factor: float
):
    threads_per_block = 256
    total_elements = batch_size * out_channels * height * width
    grid_x = (total_elements + threads_per_block - 1) // threads_per_block
    
    fused_transpose_conv_mean_bias_softmax_tanh_scale_kernel(
        mInput, mWeight, mBias, mTemp,
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_size, kernel_size, kernel_size, stride, stride, stride,
        padding, padding, padding, scaling_factor
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.jit
def softmax_channel_kernel_host(mInput: cute.Tensor, mOutput: cute.Tensor, batch_size: int, channels: int, spatial_size: int):
    threads_per_block = 256
    grid_x = (batch_size * spatial_size + threads_per_block - 1) // threads_per_block
    
    softmax_channel_kernel(mInput, mOutput, batch_size, channels, spatial_size).launch(
        grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1)
    )

@cute.jit
def tanh_scale_kernel_host(mInput: cute.Tensor, mOutput: cute.Tensor, scaling_factor: float, total_elements: int):
    threads_per_block = 256
    grid_x = (total_elements + threads_per_block - 1) // threads_per_block
    
    tanh_scale_kernel(mInput, mOutput, scaling_factor, total_elements).launch(
        grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1)
    )

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        depth = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]
        out_channels = self.conv_transpose.out_channels
        
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        temp = torch.empty(batch_size, out_channels, 1, height, width, dtype=x.dtype, device=x.device)
        output = torch.empty_like(temp)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mTemp = from_dlpack(temp, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key = (x.dtype, batch_size, in_channels, out_channels, depth, height, width)
        compiled = self.compiled.get(key)
        
        if compiled is None:
            compiled = cute.compile(fused_transpose_conv_mean_bias_kernel_host, 
                                  mInput, mWeight, mBias, mTemp,
                                  batch_size, in_channels, out_channels, depth, height, width,
                                  self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], 
                                  self.conv_transpose.padding[0], self.scaling_factor)
            self.compiled[key] = compiled
        
        compiled(mInput, mWeight, mBias, mTemp,
                batch_size, in_channels, out_channels, depth, height, width,
                self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], 
                self.conv_transpose.padding[0], self.scaling_factor)
        
        spatial_size = height * width
        softmax_compiled = cute.compile(softmax_channel_kernel_host, mTemp, mOutput, batch_size, out_channels, spatial_size)
        softmax_compiled(mTemp, mOutput, batch_size, out_channels, spatial_size)
        
        total_elements = batch_size * out_channels * spatial_size
        tanh_scale_compiled = cute.compile(tanh_scale_kernel_host, mOutput, mTemp, self.scaling_factor, total_elements)
        tanh_scale_compiled(mOutput, mTemp, self.scaling_factor, total_elements)
        
        return mTemp