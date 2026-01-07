import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_post_conv_kernel(gX: cute.Tensor, gSum: cute.Tensor, gOut: cute.Tensor):
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
    sum_val = gSum[ci, 0, 0, 0]

    # LeakyReLU
    x_val = x_val if x_val > 0.0 else 0.2 * x_val

    # Add
    x_val = x_val + sum_val

    # Clamp
    x_val = max(-1.0, min(1.0, x_val))

    # GELU
    sqrt_2 = math.sqrt(2.0)
    erf_arg = x_val / sqrt_2
    # Approximate erf using a polynomial or something, but for simplicity, use torch.erf if possible, but since it's kernel, need to implement
    # For FP32, we can use a simple approximation or compute erf.
    # Actually, in CUDA, erf is available, but in CuTe, we might need to define it.
    # For simplicity, use a Taylor series approximation for erf.
    # erf(x) ≈ (2/sqrt(pi)) * (x - x^3/3 + x^5/10 - x^7/42 + x^9/216)
    pi = 3.141592653589793
    sqrt_pi = math.sqrt(pi)
    x = erf_arg
    erf_val = (2.0 / sqrt_pi) * (x - (x**3)/3.0 + (x**5)/10.0 - (x**7)/42.0 + (x**9)/216.0)
    gelu_val = 0.5 * x_val * (1.0 + erf_val)

    gOut[bi, ci, di, hi, wi] = gelu_val

@cute.jit
def fused_post_conv_host(mX: cute.Tensor, mSum: cute.Tensor, mOut: cute.Tensor):
    batch, channels, depth, height, width = mX.shape
    total_elems = batch * channels * depth * height * width

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_post_conv_kernel(mX, mSum, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        batch, channels, depth, height, width = x.shape
        x = x.contiguous().cuda()
        sum_tensor = self.sum_tensor.contiguous().cuda()
        out = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mSum = from_dlpack(sum_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_conv_host, mX, mSum, mOut)
            self.compiled[key] = compiled

        compiled(mX, mSum, mOut)
        return out