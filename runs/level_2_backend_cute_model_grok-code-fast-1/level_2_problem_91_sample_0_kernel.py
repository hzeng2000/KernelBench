import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_softmax_bias_scale_sigmoid_kernel(gX: cute.Tensor, gBias: cute.Tensor, scale: float, gOut: cute.Tensor):
    c = cute.arch.thread_idx(0)
    b = cute.arch.block_idx(0)
    h = cute.arch.block_idx(1)
    w = cute.arch.block_idx(2)
    
    out_channels = gX.shape[1]
    
    # Shared memory for the vector
    shared_x = cute.shared_memory(float, (out_channels,))
    shared_x[c] = gX[b, c, h, w]
    cute.arch.syncthreads()
    
    # Compute max (using thread 0)
    if c == 0:
        max_val = shared_x[0]
        for i in range(1, out_channels):
            max_val = max(max_val, shared_x[i])
        shared_max = cute.shared_memory(float, (1,))
        shared_max[0] = max_val
    cute.arch.syncthreads()
    max_val = shared_max[0]
    
    # Compute sum exp
    exp_val = math.exp(shared_x[c] - max_val)
    if c == 0:
        sum_exp = 0.0
        for i in range(out_channels):
            sum_exp += math.exp(shared_x[i] - max_val)
        shared_sum = cute.shared_memory(float, (1,))
        shared_sum[0] = sum_exp
    cute.arch.syncthreads()
    sum_exp = shared_sum[0]
    
    # Compute softmax + bias + scale + sigmoid
    softmax_val = exp_val / sum_exp
    biased = softmax_val + gBias[c, 0, 0]
    scaled = biased * scale
    sigmoid_val = 1.0 / (1.0 + math.exp(-scaled))
    gOut[b, c, h, w] = sigmoid_val

@cute.jit
def fused_softmax_bias_scale_sigmoid_host(mX: cute.Tensor, mBias: cute.Tensor, scale: float, mOut: cute.Tensor):
    batch, out_channels, H, W = mX.shape
    fused_softmax_bias_scale_sigmoid_kernel(mX, mBias, scale, mOut).launch(
        grid=(batch, H, W, 1), block=(out_channels, 1, 1)
    )

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then fuses softmax, bias addition, scaling, and sigmoid into a single CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        batch, out_channels, H, W = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty_like(x)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype, out_channels)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_softmax_bias_scale_sigmoid_host, mX, mBias, self.scaling_factor, mOut)
            self.compiled[key] = compiled
        
        compiled(mX, mBias, self.scaling_factor, mOut)
        return out