import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_avgpool3d_kernel(batch_size, channels, depth, height, width, pool_kernel_size):
    out_depth = depth // pool_kernel_size
    out_height = height // pool_kernel_size
    out_width = width // pool_kernel_size
    
    @T.prim_func
    def avgpool3d_kernel(
        Input: T.Tensor((batch_size, channels, depth, height, width), "float16"),
        Output: T.Tensor((batch_size, channels, out_depth, out_height, out_width), "float16"),
    ):
        with T.Kernel(T.ceildiv(channels, 32), T.ceildiv(out_depth, 4), T.ceildiv(out_height, 4), T.ceildiv(out_width, 4), threads=128) as (c_blk, d_blk, h_blk, w_blk):
            c_start = c_blk * 32
            d_start = d_blk * 4
            h_start = h_blk * 4
            w_start = w_blk * 4
            
            for local_c, local_d, local_h, local_w in T.Parallel(32, 4, 4, 4):
                c = c_start + local_c
                d = d_start + local_d
                h = h_start + local_h
                w = w_start + local_w
                
                if c < channels and d < out_depth and h < out_height and w < out_width:
                    sum_val = T.float16(0.0)
                    for pd in range(pool_kernel_size):
                        for ph in range(pool_kernel_size):
                            for pw in range(pool_kernel_size):
                                sum_val += Input[0, c, d*pool_kernel_size+pd, h*pool_kernel_size+ph, w*pool_kernel_size+pw]
                    Output[0, c, d, h, w] = sum_val / T.float16(pool_kernel_size**3)
    
    return tilelang.compile(avgpool3d_kernel, out_idx=[1], target="cuda")


def build_conv_transpose3d_kernel(batch_size, in_channels, out_channels, in_depth, in_height, in_width, kernel_size, stride, padding, output_padding):
    out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding
    
    @T.prim_func
    def conv_transpose3d_kernel(
        Input: T.Tensor((batch_size, in_channels, in_depth, in_height, in_width), "float16"),
        Weight: T.Tensor((in_channels, out_channels, kernel_size, kernel_size, kernel_size), "float16"),
        Output: T.Tensor((batch_size, out_channels, out_depth, out_height, out_width), "float16"),
    ):
        with T.Kernel(T.ceildiv(out_channels, 32), T.ceildiv(out_depth, 4), T.ceildiv(out_height, 4), T.ceildiv(out_width, 4), threads=128) as (c_blk, d_blk, h_blk, w_blk):
            c_start = c_blk * 32
            d_start = d_blk * 4
            h_start = h_blk * 4
            w_start = w_blk * 4
            
            for local_c, local_d, local_h, local_w in T.Parallel(32, 4, 4, 4):
                oc = c_start + local_c
                od = d_start + local_d
                oh = h_start + local_h
                ow = w_start + local_w
                
                if oc < out_channels and od < out_depth and oh < out_height and ow < out_width:
                    acc = T.float16(0.0)
                    for ic in range(in_channels):
                        for kd in range(kernel_size):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    in_d = (od + padding - kd) // stride
                                    in_h = (oh + padding - kh) // stride
                                    in_w = (ow + padding - kw) // stride
                                    if (od + padding - kd) % stride == 0 and (oh + padding - kh) % stride == 0 and (ow + padding - kw) % stride == 0:
                                        if in_d >= 0 and in_d < in_depth and in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                                            acc += Input[0, ic, in_d, in_h, in_w] * Weight[ic, oc, kd, kh, kw]
                    Output[0, oc, od, oh, ow] = acc
    
    return tilelang.compile(conv_transpose3d_kernel, out_idx=[2], target="cuda")


def build_clamp_softmax_kernel(batch_size, channels, depth, height, width, clamp_min, clamp_max):
    spatial_size = depth * height * width
    
    @T.prim_func
    def clamp_softmax_kernel(
        Input: T.Tensor((batch_size, channels, spatial_size), "float16"),
        Output: T.Tensor((batch_size, channels, spatial_size), "float16"),
    ):
        with T.Kernel(T.ceildiv(channels, 32), threads=128) as (c_blk,):
            c_start = c_blk * 32
            
            for local_c in T.Parallel(32):
                c = c_start + local_c
                if c < channels:
                    # Find max for numerical stability
                    max_val = T.float16(-1e4)
                    for s in range(spatial_size):
                        val = T.max(T.min(Input[0, c, s], T.float16(clamp_max)), T.float16(clamp_min))
                        if val > max_val:
                            max_val = val
                    
                    # Compute exp and sum
                    sum_exp = T.float16(0.0)
                    for s in range(spatial_size):
                        val = T.max(T.min(Input[0, c, s], T.float16(clamp_max)), T.float16(clamp_min))
                        exp_val = T.exp(val - max_val)
                        sum_exp += exp_val
                    
                    # Normalize
                    for s in range(spatial_size):
                        val = T.max(T.min(Input[0, c, s], T.float16(clamp_max)), T.float16(clamp_min))
                        exp_val = T.exp(val - max_val)
                        Output[0, c, s] = exp_val / sum_exp
    
    return tilelang.compile(clamp_softmax_kernel, out_idx=[1], target="cuda")


def build_scale_kernel(batch_size, channels, depth, height, width):
    @T.prim_func
    def scale_kernel(
        Input: T.Tensor((batch_size, channels, depth, height, width), "float16"),
        Scale: T.Tensor((1, channels, 1, 1, 1), "float16"),
        Output: T.Tensor((batch_size, channels, depth, height, width), "float16"),
    ):
        with T.Kernel(T.ceildiv(channels, 32), T.ceildiv(depth, 4), T.ceildiv(height, 4), T.ceildiv(width, 4), threads=128) as (c_blk, d_blk, h_blk, w_blk):
            c_start = c_blk * 32
            d_start = d_blk * 4
            h_start = h_blk * 4
            w_start = w_blk * 4
            
            for local_c, local_d, local_h, local_w in T.Parallel(32, 4, 4, 4):
                c = c_start + local_c
                d = d_start + local_d
                h = h_start + local_h
                w = w_start + local_w
                
                if c < channels and d < depth and h < height and w < width:
                    Output[0, c, d, h, w] = Input[0, c, d, h, w] * Scale[0, c, 0, 0, 0]
    
    return tilelang.compile(scale_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        
        self._kernel_cache = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
    
    def _get_avgpool_kernel(self, batch_size, channels, depth, height, width):
        key = ('avgpool', batch_size, channels, depth, height, width, self.pool_kernel_size)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_avgpool3d_kernel(batch_size, channels, depth, height, width, self.pool_kernel_size)
        return self._kernel_cache[key]
    
    def _get_conv_transpose_kernel(self, batch_size, in_channels, out_channels, in_depth, in_height, in_width):
        key = ('conv_transpose', batch_size, in_channels, out_channels, in_depth, in_height, in_width, self.kernel_size, self.stride, self.padding, self.output_padding)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_conv_transpose3d_kernel(batch_size, in_channels, out_channels, in_depth, in_height, in_width, self.kernel_size, self.stride, self.padding, self.output_padding)
        return self._kernel_cache[key]
    
    def _get_clamp_softmax_kernel(self, batch_size, channels, depth, height, width):
        key = ('clamp_softmax', batch_size, channels, depth, height, width, self.clamp_min, self.clamp_max)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_clamp_softmax_kernel(batch_size, channels, depth, height, width, self.clamp_min, self.clamp_max)
        return self._kernel_cache[key]
    
    def _get_scale_kernel(self, batch_size, channels, depth, height, width):
        key = ('scale', batch_size, channels, depth, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_scale_kernel(batch_size, channels, depth, height, width)
        return self._kernel_cache[key]
    
    def forward(self, x):
        batch_size, in_channels, depth, height, width = x.shape
        
        # AvgPool3D
        x = x.half()
        kernel = self._get_avgpool_kernel(batch_size, in_channels, depth, height, width)
        x_pooled = kernel(x)
        
        # ConvTranspose3D
        in_depth_pooled = depth // self.pool_kernel_size
        in_height_pooled = height // self.pool_kernel_size
        in_width_pooled = width // self.pool_kernel_size
        kernel = self._get_conv_transpose_kernel(batch_size, in_channels, self.out_channels, in_depth_pooled, in_height_pooled, in_width_pooled)
        weight = self.conv_transpose.weight.transpose(0, 1).half().contiguous()
        x_conv = kernel(x_pooled, weight)
        
        # Clamp + Softmax
        b, c, d, h, w = x_conv.shape
        x_flat = x_conv.view(b, c, -1).contiguous()
        kernel = self._get_clamp_softmax_kernel(batch_size, c, d, h, w)
        x_softmax = kernel(x_flat)
        x_softmax = x_softmax.view(b, c, d, h, w)
        
        # Scale
        kernel = self._get_scale_kernel(batch_size, c, d, h, w)
        scale_param = self.scale.half().contiguous()
        x_scaled = kernel(x_softmax, scale_param)
        
        return x_scaled.float()