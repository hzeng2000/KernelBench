import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_leaky_multiply_leaky_kernel(gX: cute.Tensor, gMultiplier: cute.Tensor, gY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    batch, channels, depth, height, width = gX.shape
    total_elems = batch * channels * depth * height * width

    if thread_idx >= total_elems:
        return

    # Compute indices
    wi = thread_idx % width
    hi = (thread_idx // width) % height
    di = (thread_idx // (width * height)) % depth
    ci = (thread_idx // (width * height * depth)) % channels
    bi = thread_idx // (width * height * depth * channels)

    x_val = gX[bi, ci, di, hi, wi]
    mult_val = gMultiplier[ci, 0, 0, 0]

    # First LeakyReLU
    x_val = x_val if x_val > 0.0 else 0.2 * x_val
    # Multiply
    x_val = x_val * mult_val
    # Second LeakyReLU
    x_val = x_val if x_val > 0.0 else 0.2 * x_val

    gY[bi, ci, di, hi, wi] = x_val

@cute.jit
def fused_leaky_multiply_leaky_host(mX: cute.Tensor, mMultiplier: cute.Tensor, mY: cute.Tensor):
    batch, channels, depth, height, width = mX.shape
    total_elems = batch * channels * depth * height * width

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_leaky_multiply_leaky_kernel(mX, mMultiplier, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies a fused LeakyReLU + channel-wise multiply + LeakyReLU kernel, 
    and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused kernel for LeakyReLU, multiply, LeakyReLU
        batch, channels, depth, height, width = x.shape
        x = x.contiguous().cuda()
        multiplier = self.multiplier.contiguous().cuda()
        y = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mMultiplier = from_dlpack(multiplier, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_leaky_multiply_leaky_host, mX, mMultiplier, mY)
            self.compiled[key] = compiled

        compiled(mX, mMultiplier, mY)
        x = y
        x = self.max_pool(x)
        return x