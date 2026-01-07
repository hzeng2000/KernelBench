import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv_transpose_bn_mean_subtract_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth_out: int,
    height_out: int,
    width_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_D: int = 4,
    block_H: int = 8,
    block_W: int = 8,
    block_C: int = 16,
    threads: int = 256,
    dtype: str = "float16"
):
    depth_in = (depth_out + 2 * padding - kernel_size) // stride + 1
    height_in = (height_out + 2 * padding - kernel_size) // stride + 1
    width_in = (width_out + 2 * padding - kernel_size) // stride + 1

    @T.prim_func
    def kernel(
        Input: T.Tensor((batch_size, in_channels, depth_in, height_in, width_in), dtype),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype),
        Bias: T.Tensor((out_channels,), dtype),
        RunningMean: T.Tensor((out_channels,), dtype),
        RunningVar: T.Tensor((out_channels,), dtype),
        Gamma: T.Tensor((out_channels,), dtype),
        Beta: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, out_channels, depth_out, height_out, width_out), dtype),
    ):
        with T.Kernel(
            T.ceildiv(width_out, block_W),
            T.ceildiv(height_out, block_H),
            T.ceildiv(depth_out, block_D),
            batch_size,
            threads=threads
        ) as (bx, by, bz, b_n):
            # Allocate shared memory for intermediate results
            shared_mem = T.alloc_shared((block_D, block_H, block_W, block_C), dtype)
            spatial_count = depth_out * height_out * width_out
            
            for c_o in T.serial(T.ceildiv(out_channels, block_C)):
                # Compute conv transpose + BN + mean subtraction for this tile
                for d, h, w, c in T.Parallel(block_D, block_H, block_W, block_C):
                    out_d = bz * block_D + d
                    out_h = by * block_H + h
                    out_w = bx * block_W + w
                    out_c = c_o * block_C + c
                    
                    if out_d < depth_out and out_h < height_out and out_w < width_out and out_c < out_channels:
                        acc = 0.0
                        # Conv transpose computation
                        for k_c in range(in_channels):
                            for k_d in range(kernel_size):
                                for k_h in range(kernel_size):
                                    for k_w in range(kernel_size):
                                        in_d = (out_d + padding - k_d) // stride
                                        in_h = (out_h + padding - k_h) // stride
                                        in_w = (out_w + padding - k_w) // stride
                                        
                                        if (out_d + padding - k_d) % stride == 0 and \
                                           (out_h + padding - k_h) % stride == 0 and \
                                           (out_w + padding - k_w) % stride == 0 and \
                                           in_d >= 0 and in_d < depth_in and \
                                           in_h >= 0 and in_h < height_in and \
                                           in_w >= 0 and in_w < width_in:
                                            acc += Input[b_n, k_c, in_d, in_h, in_w] * Weight[k_c, out_c, k_d, k_h, k_w]
                        
                        # Add bias
                        acc += Bias[out_c]
                        
                        # Batch normalization (using running stats)
                        bn_val = (acc - RunningMean[out_c]) / T.sqrt(RunningVar[out_c] + 1e-5)
                        bn_val = bn_val * Gamma[out_c] + Beta[out_c]
                        
                        shared_mem[d, h, w, c] = bn_val
                    else:
                        shared_mem[d, h, w, c] = 0.0
                
                # Compute mean across spatial dimensions for this channel tile
                for c in range(block_C):
                    ch = c_o * block_C + c
                    if ch < out_channels:
                        sum_val = 0.0
                        for d in range(block_D):
                            for h in range(block_H):
                                for w in range(block_W):
                                    sum_val += shared_mem[d, h, w, c]
                        
                        # Subtract mean
                        mean_val = sum_val / spatial_count
                        for d in range(block_D):
                            for h in range(block_H):
                                for w in range(block_W):
                                    out_d = bz * block_D + d
                                    out_h = by * block_H + h
                                    out_w = bx * block_W + w
                                    if out_d < depth_out and out_h < height_out and out_w < width_out:
                                        Output[b_n, ch, out_d, out_h, out_w] = shared_mem[d, h, w, c] - mean_val

    return tilelang.compile(kernel, out_idx=[6], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self._kernel_cache = {}
        
    def _get_output_shape(self, depth_in, height_in, width_in):
        depth_out = (depth_in - 1) * self.conv_transpose.stride[0] - 2 * self.conv_transpose.padding[0] + self.conv_transpose.kernel_size[0]
        height_out = (height_in - 1) * self.conv_transpose.stride[1] - 2 * self.conv_transpose.padding[1] + self.conv_transpose.kernel_size[1]
        width_out = (width_in - 1) * self.conv_transpose.stride[2] - 2 * self.conv_transpose.padding[2] + self.conv_transpose.kernel_size[2]
        return depth_out, height_out, width_out

    def _get_kernel(self, batch_size, in_channels, out_channels, depth_out, height_out, width_out, kernel_size, stride, padding):
        key = (batch_size, in_channels, out_channels, depth_out, height_out, width_out, kernel_size, stride, padding)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose_bn_mean_subtract_kernel(
                batch_size, in_channels, out_channels, depth_out, height_out, width_out,
                kernel_size, stride, padding, dtype="float16"
            )
        return self._kernel_cache[key]

    def forward(self, x):
        # Get shapes
        batch_size, _, depth_in, height_in, width_in = x.shape
        depth_out, height_out, width_out = self._get_output_shape(depth_in, height_in, width_in)
        
        # Ensure input is contiguous and in fp16
        x = x.contiguous().half()
        
        # Get kernel
        kernel = self._get_kernel(
            batch_size, self.conv_transpose.in_channels, self.conv_transpose.out_channels,
            depth_out, height_out, width_out,
            self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], self.conv_transpose.padding[0]
        )
        
        # Get weights and bias
        weight = self.conv_transpose.weight.data.half()
        bias = self.conv_transpose.bias.data.half() if self.conv_transpose.bias is not None else torch.zeros(self.conv_transpose.out_channels, device=x.device, dtype=torch.float16)
        
        # Get batch norm parameters
        running_mean = self.batch_norm.running_mean.data.half()
        running_var = self.batch_norm.running_var.data.half()
        gamma = self.batch_norm.weight.data.half()
        beta = self.batch_norm.bias.data.half()
        
        # Run fused kernel
        output = kernel(x, weight, bias, running_mean, running_var, gamma, beta)
        
        return output