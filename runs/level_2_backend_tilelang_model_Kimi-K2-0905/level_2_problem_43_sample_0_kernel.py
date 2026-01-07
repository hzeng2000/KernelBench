import torch
import torch.nn as nn
import tilelang
import tilelang.language as T
import math


def build_fused_conv3d_maxpool_logsumexp_relu_kernel(
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    pool_kernel: int = 2,
    pool_stride: int = 2,
    block_d: int = 4,
    block_h: int = 8,
    block_w: int = 8,
    threads: int = 256,
    dtype: str = "float16"
):
    # Output dimensions after conv3d
    out_d = (depth + 2 * padding - kernel_size) // stride + 1
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1
    
    # Output dimensions after maxpool3d
    pool_out_d = out_d // pool_stride
    pool_out_h = out_h // pool_stride
    pool_out_w = out_w // pool_stride
    
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_channels, depth, height, width), dtype),
        W: T.Tensor((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype),
        B: T.Tensor((out_channels,), dtype),
        Output: T.Tensor((batch_size, 1, pool_out_d, pool_out_h, pool_out_w), dtype),
    ):
        # Shared memory for convolution input tile
        shared_X = T.alloc_shared((block_d + 2, block_h + 2, block_w + 2, in_channels), dtype)
        # Shared memory for weights
        shared_W = T.alloc_shared((out_channels, in_channels, kernel_size, kernel_size, kernel_size), dtype)
        # Register for convolution output
        reg_conv = T.alloc_fragment((block_d, block_h, block_w), dtype)
        # Register for maxpool output
        reg_pool = T.alloc_fragment((block_d // pool_stride, block_h // pool_stride, block_w // pool_stride), dtype)
        # Register for logsumexp accumulation
        reg_logsumexp = T.alloc_fragment((1,), "float32")
        
        with T.Kernel(T.ceildiv(pool_out_w, block_w), T.ceildiv(pool_out_h, block_h), 
                     T.ceildiv(pool_out_d, block_d), batch_size, threads=threads) as (bx, by, bz, batch):
            
            # Load weights to shared memory (coalesced)
            for oc in T.Parallel(out_channels):
                for ic in range(in_channels):
                    for kz in range(kernel_size):
                        for ky in range(kernel_size):
                            for kx in range(kernel_size):
                                shared_W[oc, ic, kz, ky, kx] = W[oc, ic, kz, ky, kx]
            
            # Iterate over output spatial dimensions
            for od in T.serial(T.ceildiv(pool_out_d, block_d)):
                for oh in T.serial(T.ceildiv(pool_out_h, block_h)):
                    for ow in T.serial(T.ceildiv(pool_out_w, block_w)):
                        
                        # Clear logsumexp accumulator
                        reg_logsumexp[0] = T.cast(-1e9, "float32")
                        
                        # Iterate over output channels for logsumexp
                        for oc in range(out_channels):
                            
                            # Clear convolution output registers
                            for tz in T.Parallel(block_d):
                                for ty in range(block_h):
                                    for tx in range(block_w):
                                        reg_conv[tz, ty, tx] = T.cast(0.0, dtype)
                            
                            # Load input tile to shared memory with halo
                            for tz in T.Parallel(block_d + 2):
                                for ty in range(block_h + 2):
                                    for tx in range(block_w + 2):
                                        for ic in range(in_channels):
                                            in_z = od * block_d * pool_stride + tz * stride - padding
                                            in_y = oh * block_h * pool_stride + ty * stride - padding
                                            in_x = ow * block_w * pool_stride + tx * stride - padding
                                            
                                            if (in_z >= 0 and in_z < depth and 
                                                in_y >= 0 and in_y < height and 
                                                in_x >= 0 and in_x < width):
                                                shared_X[tz, ty, tx, ic] = X[batch, ic, in_z, in_y, in_x]
                                            else:
                                                shared_X[tz, ty, tx, ic] = T.cast(0.0, dtype)
                            
                            # Perform 3D convolution
                            for tz in T.Parallel(block_d):
                                for ty in range(block_h):
                                    for tx in range(block_w):
                                        for ic in range(in_channels):
                                            for kz in range(kernel_size):
                                                for ky in range(kernel_size):
                                                    for kx in range(kernel_size):
                                                        reg_conv[tz, ty, tx] += (
                                                            shared_X[tz * pool_stride + kz, 
                                                                   ty * pool_stride + ky, 
                                                                   tx * pool_stride + kx, ic] * 
                                                            shared_W[oc, ic, kz, ky, kx]
                                                        )
                                        # Add bias
                                        reg_conv[tz, ty, tx] += B[oc]
                            
                            # Perform max pooling (2x2x2) with stride 2
                            for tz in T.Parallel(block_d // pool_stride):
                                for ty in range(block_h // pool_stride):
                                    for tx in range(block_w // pool_stride):
                                        max_val = reg_conv[tz * pool_stride, ty * pool_stride, tx * pool_stride]
                                        for pz in range(pool_kernel):
                                            for py in range(pool_kernel):
                                                for px in range(pool_kernel):
                                                    val = reg_conv[tz * pool_stride + pz, 
                                                                 ty * pool_stride + py, 
                                                                 tx * pool_stride + px]
                                                    max_val = T.max(max_val, val)
                                        reg_pool[tz, ty, tx] = max_val
                            
                            # Update logsumexp
                            for tz in T.Parallel(block_d // pool_stride):
                                for ty in range(block_h // pool_stride):
                                    for tx in range(block_w // pool_stride):
                                        out_z = od * (block_d // pool_stride) + tz
                                        out_y = oh * (block_h // pool_stride) + ty
                                        out_x = ow * (block_w // pool_stride) + tx
                                        
                                        if (out_z < pool_out_d and out_y < pool_out_h and out_x < pool_out_w):
                                            val = T.cast(reg_pool[tz, ty, tx], "float32")
                                            # Online logsumexp computation
                                            if reg_logsumexp[0] == T.cast(-1e9, "float32"):
                                                reg_logsumexp[0] = val
                                            else:
                                                max_prev = reg_logsumexp[0]
                                                reg_logsumexp[0] = max_prev + T.log1p(T.exp(val - max_prev))
                        
                        # Apply ReLU and store final output
                        for tz in T.Parallel(block_d // pool_stride):
                            for ty in range(block_h // pool_stride):
                                for tx in range(block_w // pool_stride):
                                    out_z = od * (block_d // pool_stride) + tz
                                    out_y = oh * (block_h // pool_stride) + ty
                                    out_x = ow * (block_w // pool_stride) + tx
                                    
                                    if (out_z < pool_out_d and out_y < pool_out_h and out_x < pool_out_w):
                                        final_val = T.max(T.cast(reg_logsumexp[0], dtype), T.cast(0.0, dtype))
                                        Output[batch, 0, out_z, out_y, out_x] = final_val

    return tilelang.compile(fused_kernel, out_idx=[3], target="cuda")


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self._kernel_cache = {}
        
        # Store dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def _get_kernel(self, batch_size: int, depth: int, height: int, width: int):
        key = (batch_size, depth, height, width)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_fused_conv3d_maxpool_logsumexp_relu_kernel(
                batch_size=batch_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                depth=depth,
                height=height,
                width=width,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
        return self._kernel_cache[key]

    def forward(self, x):
        batch_size, _, depth, height, width = x.shape
        
        # Get kernel
        kernel = self._get_kernel(batch_size, depth, height, width)
        
        # Get weights and bias from conv layer
        weight = self.conv.weight.half()
        bias = self.conv.bias.half()
        
        # Allocate output tensor
        out_d = ((depth + 2 * self.padding - self.kernel_size) // self.stride + 1) // 2
        out_h = ((height + 2 * self.padding - self.kernel_size) // self.stride + 1) // 2
        out_w = ((width + 2 * self.padding - self.kernel_size) // self.stride + 1) // 2
        output = torch.empty(batch_size, 1, out_d, out_h, out_w, dtype=torch.float16, device=x.device)
        
        # Run fused kernel
        kernel(x.half(), weight, bias, output)
        
        return output