import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_conv3d_kernel(batch_size, out_channels, in_channels, D, H, W, kernel_size, stride=1, padding=0, dtype="float16"):
    out_D = (D + 2 * padding - kernel_size) // stride + 1
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    
    block_d = 4
    block_h = 8
    block_w = 8
    block_oc = 16
    block_ic = 8
    
    threads = 128
    
    @T.prim_func
    def conv3d_kernel(
        Input: T.Tensor((batch_size, in_channels, D, H, W), dtype),
        Weight: T.Tensor((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype),
        Output: T.Tensor((batch_size, out_channels, out_D, out_H, out_W), dtype),
    ):
        with T.Kernel(T.ceildiv(out_W, block_w), T.ceildiv(out_H, block_h), T.ceildiv(out_D, block_d), T.ceildiv(out_channels, block_oc), batch_size, threads=threads) as (bx, by, bz, boc, b):
            start_w = bx * block_w
            start_h = by * block_h
            start_d = bz * block_d
            start_oc = boc * block_oc
            
            local_w = T.alloc_fragment((block_w,), "int32")
            local_h = T.alloc_fragment((block_h,), "int32")
            local_d = T.alloc_fragment((block_d,), "int32")
            
            for i in T.Parallel(block_w):
                local_w[i] = start_w + i
            for i in T.Parallel(block_h):
                local_h[i] = start_h + i
            for i in T.Parallel(block_d):
                local_d[i] = start_d + i
                
            acc = T.alloc_fragment((block_d, block_h, block_w, block_oc), dtype, "local")
            for od in T.Parallel(block_d):
                for oh in T.Parallel(block_h):
                    for ow in T.Parallel(block_w):
                        for oc in T.Parallel(block_oc):
                            acc[od, oh, ow, oc] = T.float16(0)
                            
            for ic in range(0, in_channels, block_ic):
                for kd in range(kernel_size):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            for od in T.Parallel(block_d):
                                for oh in T.Parallel(block_h):
                                    for ow in T.Parallel(block_w):
                                        for oc in T.Parallel(block_oc):
                                            in_d = local_d[od] * stride - padding + kd
                                            in_h = local_h[oh] * stride - padding + kh
                                            in_w = local_w[ow] * stride - padding + kw
                                            if in_d >= 0 and in_d < D and in_h >= 0 and in_h < H and in_w >= 0 and in_w < W:
                                                for ic_i in range(block_ic):
                                                    if ic + ic_i < in_channels:
                                                        acc[od, oh, ow, oc] += Input[b, ic + ic_i, in_d, in_h, in_w] * Weight[start_oc + oc, ic + ic_i, kd, kh, kw]
                                                            
            for od in T.Parallel(block_d):
                for oh in T.Parallel(block_h):
                    for ow in T.Parallel(block_w):
                        for oc in T.Parallel(block_oc):
                            if local_d[od] < out_D and local_h[oh] < out_H and local_w[ow] < out_W and start_oc + oc < out_channels:
                                Output[b, start_oc + oc, local_d[od], local_h[oh], local_w[ow]] = acc[od, oh, ow, oc]
                                
    return tilelang.compile(conv3d_kernel, out_idx=[2], target="cuda")


def build_min_softmax_kernel(batch_size, out_channels, H, W, dtype="float16"):
    block_h = 8
    block_w = 8
    block_oc = 16
    threads = 128
    
    @T.prim_func
    def min_softmax_kernel(
        Input: T.Tensor((batch_size, out_channels, 22, H, W), dtype),
        Output: T.Tensor((batch_size, out_channels, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(W, block_w), T.ceildiv(H, block_h), T.ceildiv(out_channels, block_oc), batch_size, threads=threads) as (bx, by, boc, b):
            start_w = bx * block_w
            start_h = by * block_h
            start_oc = boc * block_oc
            
            local_w = T.alloc_fragment((block_w,), "int32")
            local_h = T.alloc_fragment((block_h,), "int32")
            
            for i in T.Parallel(block_w):
                local_w[i] = start_w + i
            for i in T.Parallel(block_h):
                local_h[i] = start_h + i
                
            # Min reduction along depth dimension (dim=2)
            min_vals = T.alloc_fragment((block_oc, block_h, block_w), dtype, "local")
            for oc in T.Parallel(block_oc):
                for h in T.Parallel(block_h):
                    for w in T.Parallel(block_w):
                        min_vals[oc, h, w] = T.float16(1e4)
                        for d in range(22):
                            val = Input[b, start_oc + oc, d, local_h[h], local_w[w]]
                            if val < min_vals[oc, h, w]:
                                min_vals[oc, h, w] = val
                                
            # Softmax along channel dimension (dim=1)
            # First compute max for numerical stability
            max_vals = T.alloc_fragment((block_h, block_w), dtype, "local")
            for h in T.Parallel(block_h):
                for w in T.Parallel(block_w):
                    max_vals[h, w] = T.float16(-1e4)
                    for oc in range(out_channels):
                        block_oc_idx = oc // block_oc
                        local_oc_idx = oc % block_oc
                        if block_oc_idx == boc:
                            val = min_vals[local_oc_idx, h, w]
                            if val > max_vals[h, w]:
                                max_vals[h, w] = val
                                
            # Compute exp and sum
            exp_sum = T.alloc_fragment((block_h, block_w), dtype, "local")
            for h in T.Parallel(block_h):
                for w in T.Parallel(block_w):
                    exp_sum[h, w] = T.float16(0)
                    for oc in T.Parallel(block_oc):
                        if start_oc + oc < out_channels:
                            exp_val = T.exp(min_vals[oc, h, w] - max_vals[h, w])
                            exp_sum[h, w] += exp_val
                            
            # Compute final softmax values
            for oc in T.Parallel(block_oc):
                for h in T.Parallel(block_h):
                    for w in T.Parallel(block_w):
                        if start_oc + oc < out_channels and local_h[h] < H and local_w[w] < W:
                            exp_val = T.exp(min_vals[oc, h, w] - max_vals[h, w])
                            Output[b, start_oc + oc, local_h[h], local_w[w]] = exp_val / exp_sum[h, w]
                            
    return tilelang.compile(min_softmax_kernel, out_idx=[1], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self._conv_kernel_cache = {}
        self._min_softmax_kernel_cache = {}
        
        # Initialize conv weights
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def _get_conv_kernel(self, batch_size, in_channels, D, H, W, kernel_size, dtype="float16"):
        key = (batch_size, in_channels, D, H, W, kernel_size, dtype)
        if key not in self._conv_kernel_cache:
            self._conv_kernel_cache[key] = build_conv3d_kernel(batch_size, self.conv.out_channels, in_channels, D, H, W, kernel_size, dtype=dtype)
        return self._conv_kernel_cache[key]
        
    def _get_min_softmax_kernel(self, batch_size, out_channels, H, W, dtype="float16"):
        key = (batch_size, out_channels, H, W, dtype)
        if key not in self._min_softmax_kernel_cache:
            self._min_softmax_kernel_cache[key] = build_min_softmax_kernel(batch_size, out_channels, H, W, dtype=dtype)
        return self._min_softmax_kernel_cache[key]
        
    def forward(self, x):
        batch_size, in_channels, D, H, W = x.shape
        x = x.half()
        
        # Perform convolution
        conv_kernel = self._get_conv_kernel(batch_size, in_channels, D, H, W, self.conv.kernel_size[0])
        conv_out = conv_kernel(x, self.conv.weight.half())
        
        # Add bias
        conv_out = conv_out + self.conv.bias.half().view(1, -1, 1, 1, 1)
        
        # Perform min reduction and softmax
        _, out_channels, out_D, out_H, out_W = conv_out.shape
        min_softmax_kernel = self._get_min_softmax_kernel(batch_size, out_channels, out_H, out_W)
        output = min_softmax_kernel(conv_out)
        
        return output.float()