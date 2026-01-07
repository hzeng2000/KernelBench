import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_post_kernel(gX: cute.Tensor, gBias: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    batch, out_c, d, h, w = gX.shape
    total_elems = batch * out_c * d * h * w

    if thread_idx >= total_elems:
        return

    b = thread_idx // (out_c * d * h * w)
    remainder = thread_idx % (out_c * d * h * w)
    c = remainder // (d * h * w)
    remainder2 = remainder % (d * h * w)
    dd = remainder2 // (h * w)
    remainder3 = remainder2 % (h * w)
    hh = remainder3 // w
    ww = remainder3 % w

    val = gX[b, c, dd, hh, ww]

    # ReLU
    val = cute.max(0.0, val)

    # LeakyReLU
    val = cute.where(val > 0.0, val, 0.01 * val)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = 0.7978845608028654  # sqrt(2/pi)
    gelu_arg = sqrt_2_pi * (val + 0.044715 * val * val * val)
    val = 0.5 * val * (1.0 + cute.tanh(gelu_arg))

    # Sigmoid
    val = 1.0 / (1.0 + cute.exp(-val))

    # Add bias
    val = val + gBias[c, 0, 0, 0]

    gC[b, c, dd, hh, ww] = val

@cute.jit
def fused_post_host(mX: cute.Tensor, mBias: cute.Tensor, mC: cute.Tensor):
    batch, out_c, d, h, w = mX.shape
    total_elems = batch * out_c * d * h * w

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_post_kernel(mX, mBias, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs 3D convolution with PyTorch, then applies fused ReLU, LeakyReLU, GELU, Sigmoid, and bias addition in a custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        batch, out_c, d, h, w = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        C = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_host, mX, mBias, mC)
            self.compiled[key] = compiled

        compiled(mX, mBias, mC)
        return C