import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def min_kernel(gA: cute.Tensor, gTemp: cute.Tensor): 
    tidx = cute.arch.thread_idx().x  
    bidx = cute.arch.block_idx().x  
    bdim = cute.arch.block_dim().x  

    thread_idx = bidx * bdim + tidx

    B, C, H, W = gA.shape
    total = B * H * W
    if thread_idx >= total:
        return

    b = thread_idx // (H * W)
    hw = thread_idx % (H * W)
    h = hw // W
    w = hw % W

    min_val = float('inf')
    for c in range(C):
        val = gA[b, c, h, w]
        if val < min_val:
            min_val = val
    gTemp[b, h, w] = min_val

@cute.jit
def min_host(mA: cute.Tensor, mTemp: cute.Tensor):
    B, C, H, W = mA.shape
    total = B * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    min_kernel(mA, mTemp).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def sum_kernel(gTemp: cute.Tensor, gOut: cute.Tensor): 
    tidx = cute.arch.thread_idx().x  
    bidx = cute.arch.block_idx().x  
    bdim = cute.arch.block_dim().x  

    thread_idx = bidx * bdim + tidx

    B, H, W = gTemp.shape
    total = B * W
    if thread_idx >= total:
        return

    b = thread_idx // W
    w = thread_idx % W

    sum_val = 0.0
    for h in range(H):
        sum_val += gTemp[b, h, w]
    gOut[b, 0, 0, w] = sum_val

@cute.jit
def sum_host(mTemp: cute.Tensor, mOut: cute.Tensor):
    B, H, W = mTemp.shape
    total = B * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    sum_kernel(mTemp, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def gelu_add_kernel(gOut: cute.Tensor, gBias: cute.Tensor, gFinal: cute.Tensor): 
    tidx = cute.arch.thread_idx().x  
    bidx = cute.arch.block_idx().x  
    bdim = cute.arch.block_dim().x  

    thread_idx = bidx * bdim + tidx

    B, _, _, W = gOut.shape
    total = B * W
    if thread_idx >= total:
        return

    b = thread_idx // W
    w = thread_idx % W

    x = gOut[b, 0, 0, w]
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    x_cubed = x * x * x
    inner = x + 0.044715 * x_cubed
    tanh_inner = sqrt_2_pi * inner
    tanh_val = math.tanh(tanh_inner)
    gelu_val = 0.5 * x * (1.0 + tanh_val)
    gFinal[b, 0, 0, w] = gelu_val + gBias[0]

@cute.jit
def gelu_add_host(mOut: cute.Tensor, mBias: cute.Tensor, mFinal: cute.Tensor):
    B, _, _, W = mOut.shape
    total = B * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total, threads_per_block)
    gelu_add_kernel(mOut, mBias, mFinal).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        temp = torch.empty((B, H, W), dtype=x.dtype, device=x.device).contiguous()
        out = torch.empty((B, 1, 1, W), dtype=x.dtype, device=x.device).contiguous()
        final = torch.empty((B, 1, 1, W), dtype=x.dtype, device=x.device).contiguous()

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mTemp = from_dlpack(temp, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mFinal = from_dlpack(final, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled_min = self.compiled.get('min', {}).get(key)
        if compiled_min is None:
            compiled_min = cute.compile(min_host, mX, mTemp)
            self.compiled.setdefault('min', {})[key] = compiled_min
        compiled_min(mX, mTemp)

        compiled_sum = self.compiled.get('sum', {}).get(key)
        if compiled_sum is None:
            compiled_sum = cute.compile(sum_host, mTemp, mOut)
            self.compiled.setdefault('sum', {})[key] = compiled_sum
        compiled_sum(mTemp, mOut)

        compiled_gelu = self.compiled.get('gelu', {}).get(key)
        if compiled_gelu is None:
            compiled_gelu = cute.compile(gelu_add_host, mOut, mBias, mFinal)
            self.compiled.setdefault('gelu', {})[key] = compiled_gelu
        compiled_gelu(mOut, mBias, mFinal)
        return final