import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_gelu_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    height: int, width: int, kernel_size: int
):
    # Shared memory for input tile
    shared_x = cute.shared_tensor((16, 16, 16), dtype=cute.float32)
    shared_w = cute.shared_tensor((16, 16, 9), dtype=cute.float32)
    
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    out_c = bidz * 16 + tidy
    batch = bidx
    out_h = bidy * 8 + tidx // 4
    out_w = (tidx % 4) * 2
    
    if out_c < out_channels and batch < batch_size and out_h < height and out_w < width:
        acc = 0.0
        for ic in range(0, in_channels, 16):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_h = out_h + kh - 1
                    in_w = out_w + kw - 1
                    if in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                        for ti in range(16):
                            if ic + ti < in_channels:
                                x_val = gX[batch, ic + ti, in_h, in_w]
                                w_val = gW[out_c, ic + ti, kh, kw]
                                acc += x_val * w_val
        
        # Add bias
        if out_c < out_channels:
            acc += gB[out_c]
        
        # GELU activation
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = acc * acc * acc
        tanh_arg = 0.7978845608 * (acc + 0.044715 * x_cubed)
            # Approximate tanh
        tanh_val = tanh_arg / (1.0 + abs(tanh_arg))
        gelu_val = 0.5 * acc * (1.0 + tanh_val)
        
        gY[batch, out_c, out_h, out_w] = gelu_val

@cute.kernel
def global_avg_pool_kernel(
    gX: cute.Tensor, gY: cute.Tensor,
    batch_size: int, channels: int, height: int, width: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    
    batch = bidx
    ch = bidy * 32 + tidy
    
    if batch < batch_size and ch < channels:
        sum_val = 0.0
        for h in range(height):
            for w in range(width):
                sum_val += gX[batch, ch, h, w]
        
        avg_val = sum_val / (height * width)
        gY[batch, ch] = avg_val

@cute.jit
def conv_gelu_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    height: int, width: int, kernel_size: int
):
    grid_x = batch_size
    grid_y = (height + 7) // 8
    grid_z = (out_channels + 15) // 16
    
    conv_gelu_kernel(mX, mW, mB, mY, batch_size, in_channels, out_channels, height, width, kernel_size).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(32, 16, 1)
    )

@cute.jit
def global_avg_pool_host(
    mX: cute.Tensor, mY: cute.Tensor,
    batch_size: int, channels: int, height: int, width: int
):
    grid_x = batch_size
    grid_y = (channels + 31) // 32
    grid_z = 1
    
    global_avg_pool_kernel(mX, mY, batch_size, channels, height, width).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(1, 32, 1)
    )

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.compiled = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]
        
        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()
        
        # Conv + GELU output
        conv_out = torch.empty((batch_size, out_channels, height, width), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype, "conv_gelu")
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_gelu_host, mX, mW, mB, mConvOut, batch_size, in_channels, out_channels, height, width, kernel_size)
            self.compiled[key] = compiled
        
        compiled(mX, mW, mB, mConvOut, batch_size, in_channels, out_channels, height, width, kernel_size)
        
        # Global average pooling
        output = torch.empty((batch_size, out_channels), dtype=x.dtype, device=x.device)
        
        mPoolOut = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key_pool = (x.dtype, "global_avg_pool")
        compiled_pool = self.compiled.get(key_pool)
        if compiled_pool is None:
            compiled_pool = cute.compile(global_avg_pool_host, mConvOut, mPoolOut, batch_size, out_channels, height, width)
            self.compiled[key_pool] = compiled_pool
        
        compiled_pool(mConvOut, mPoolOut, batch_size, out_channels, height, width)
        
        return output