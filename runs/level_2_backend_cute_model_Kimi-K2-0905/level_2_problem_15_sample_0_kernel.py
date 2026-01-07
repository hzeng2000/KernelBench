import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose_bn_mean_sub_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gRunningMean: cute.Tensor, gRunningVar: cute.Tensor,
    gScale: cute.Tensor, gBias: cute.Tensor,
    gOut: cute.Tensor,
    batch_size: int, out_channels: int, out_depth: int, out_height: int, out_width: int,
    in_channels: int, in_depth: int, in_height: int, in_width: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    thread_idx = (bidz * cute.arch.grid_dim().y * cute.arch.grid_dim().x +
                  bidy * cute.arch.grid_dim().x + bidx) * (bdimx * bdimy * bdimz) +
                  tidz * bdimx * bdimy + tidy * bdimx + tidx
    
    total_threads = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z * bdimx * bdimy * bdimz
    
    total_output_elements = batch_size * out_channels * out_depth * out_height * out_width
    
    for idx in range(thread_idx, total_output_elements, total_threads):
        n = idx // (out_channels * out_depth * out_height * out_width)
        rem = idx % (out_channels * out_depth * out_height * out_width)
        c = rem // (out_depth * out_height * out_width)
        rem = rem % (out_depth * out_height * out_width)
        d = rem // (out_height * out_width)
        rem = rem % (out_height * out_width)
        h = rem // out_width
        w = rem % out_width
        
        in_d_start = d * stride_d - pad_d
        in_h_start = h * stride_h - pad_h
        in_w_start = w * stride_w - pad_w
        
        out_val = 0.0
        if gB is not None:
            out_val = gB[c]
        
        for ic in range(in_channels):
            for kd in range(kernel_d):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        in_d = in_d_start + kd
                        in_h = in_h_start + kh
                        in_w = in_w_start + kw
                        
                        if in_d >= 0 and in_d < in_depth and in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                            weight_idx = c * in_channels * kernel_d * kernel_h * kernel_w + ic * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw
                            input_idx = n * in_channels * in_depth * in_height * in_width + ic * in_depth * in_height * in_width + in_d * in_height * in_width + in_h * in_width + in_w
                            out_val += gX[input_idx] * gW[weight_idx]
        
        # BatchNorm
        bn_idx = c
        mean = gRunningMean[bn_idx]
        var = gRunningVar[bn_idx]
        scale = gScale[bn_idx]
        bias = gBias[bn_idx]
        
        bn_out = (out_val - mean) / cute.math.sqrt(var + 1e-5)
        bn_out = bn_out * scale + bias
        
        gOut[idx] = bn_out

@cute.kernel
def compute_mean_kernel(
    gX: cute.Tensor, gMean: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int,
    spatial_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    thread_idx = (bidz * cute.arch.grid_dim().y * cute.arch.grid_dim().x +
                  bidy * cute.arch.grid_dim().x + bidx) * (bdimx * bdimy * bdimz) +
                  tidz * bdimx * bdimy + tidy * bdimx + tidx
    
    total_threads = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z * bdimx * bdimy * bdimz
    
    for n in range(batch_size):
        for c in range(channels):
            idx = n * channels + c
            if idx % total_threads == thread_idx:
                sum_val = 0.0
                for d in range(depth):
                    for h in range(height):
                        for w in range(width):
                            x_idx = n * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w
                            sum_val += gX[x_idx]
                gMean[n * channels + c] = sum_val / spatial_size

@cute.kernel
def subtract_mean_kernel(
    gX: cute.Tensor, gMean: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    thread_idx = (bidz * cute.arch.grid_dim().y * cute.arch.grid_dim().x +
                  bidy * cute.arch.grid_dim().x + bidx) * (bdimx * bdimy * bdimz) +
                  tidz * bdimx * bdimy + tidy * bdimx + tidx
    
    total_elements = batch_size * channels * depth * height * width
    total_threads = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z * bdimx * bdimy * bdimz
    
    for idx in range(thread_idx, total_elements, total_threads):
        n = idx // (channels * depth * height * width)
        rem = idx % (channels * depth * height * width)
        c = rem // (depth * height * width)
        
        mean_idx = n * channels + c
        mean_val = gMean[mean_idx]
        
        gX[idx] = gX[idx] - mean_val

@cute.jit
def conv_transpose_bn_mean_sub_fused_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mRunningMean: cute.Tensor, mRunningVar: cute.Tensor,
    mScale: cute.Tensor, mBias: cute.Tensor,
    mOut: cute.Tensor,
    batch_size: int, out_channels: int, out_depth: int, out_height: int, out_width: int,
    in_channels: int, in_depth: int, in_height: int, in_width: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    threads_per_block = 256
    total_output_elements = batch_size * out_channels * out_depth * out_height * out_width
    grid_x = cute.ceil_div(total_output_elements, threads_per_block)
    
    conv_transpose_bn_mean_sub_kernel(
        mX, mW, mB, mRunningMean, mRunningVar, mScale, mBias, mOut,
        batch_size, out_channels, out_depth, out_height, out_width,
        in_channels, in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.jit
def compute_and_subtract_mean_host(
    mX: cute.Tensor, mMean: cute.Tensor,
    batch_size: int, channels: int, depth: int, height: int, width: int,
    spatial_size: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(batch_size * channels, threads_per_block)
    
    compute_mean_kernel(mX, mMean, batch_size, channels, depth, height, width, spatial_size).launch(
        grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))
    
    total_elements = batch_size * channels * depth * height * width
    grid_x2 = cute.ceil_div(total_elements, threads_per_block)
    subtract_mean_kernel(mX, mMean, batch_size, channels, depth, height, width).launch(
        grid=(grid_x2, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.bias = bias
        
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
        
        # Compute output dimensions
        out_depth = (in_depth - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        out_height = (in_height - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
        out_width = (in_width - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2]
        
        x = x.contiguous().cuda()
        
        # Get weights and biases
        weight = self.conv_transpose.weight.data.contiguous()
        bias = self.conv_transpose.bias.data.contiguous() if self.bias else None
        running_mean = self.batch_norm.running_mean.data.contiguous()
        running_var = self.batch_norm.running_var.data.contiguous()
        scale = self.batch_norm.weight.data.contiguous()
        bn_bias = self.batch_norm.bias.data.contiguous()
        
        # Output tensor
        out = torch.empty(batch_size, self.out_channels, out_depth, out_height, out_width, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,)) if self.bias else None
        mRunningMean = from_dlpack(running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningVar = from_dlpack(running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mScale = from_dlpack(scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBnBias = from_dlpack(bn_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        # Compile and run fused kernel
        key = (x.dtype, batch_size, self.out_channels, out_depth, out_height, out_width)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_transpose_bn_mean_sub_fused_host,
                mX, mW, mB, mRunningMean, mRunningVar, mScale, mBnBias, mOut,
                batch_size, self.out_channels, out_depth, out_height, out_width,
                self.in_channels, in_depth, in_height, in_width,
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.stride[0], self.stride[1], self.stride[2],
                self.padding[0], self.padding[1], self.padding[2]
            )
            self.compiled[key] = compiled
        
        compiled(
            mX, mW, mB, mRunningMean, mRunningVar, mScale, mBnBias, mOut,
            batch_size, self.out_channels, out_depth, out_height, out_width,
            self.in_channels, in_depth, in_height, in_width,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2]
        )
        
        # Compute and subtract mean
        spatial_size = out_depth * out_height * out_width
        mean_tensor = torch.empty(batch_size * self.out_channels, dtype=out.dtype, device=out.device)
        mMean = from_dlpack(mean_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut2 = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        key2 = (x.dtype, batch_size, self.out_channels, out_depth, out_height, out_width, spatial_size)
        compiled2 = self.compiled.get(key2)
        if compiled2 is None:
            compiled2 = cute.compile(
                compute_and_subtract_mean_host,
                mOut2, mMean, batch_size, self.out_channels, out_depth, out_height, out_width, spatial_size
            )
            self.compiled[key2] = compiled2
        
        compiled2(mOut2, mMean, batch_size, self.out_channels, out_depth, out_height, out_width, spatial_size)
        
        return out