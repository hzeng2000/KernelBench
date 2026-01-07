import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_post_kernel(gA: cute.Tensor, gC: cute.Tensor, add_value: float, scale: float):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    batch, out_c, H, W = gA.shape
    total_elems = batch * out_c * H * W

    if thread_idx >= total_elems:
        return

    b = thread_idx // (out_c * H * W)
    c = (thread_idx % (out_c * H * W)) // (H * W)
    h = (thread_idx % (H * W)) // W
    w = thread_idx % W

    x = gA[b, c, h, w]

    # Compute mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)), but handle large x to avoid overflow
    if x > 20.0:
        softplus_x = x
    else:
        softplus_x = cute.log(1.0 + cute.exp(x))
    tanh_softplus = cute.tanh(softplus_x)
    mish_x = x * tanh_softplus

    # Add value
    y = mish_x + add_value

    # Hardtanh: clamp to [-1, 1]
    y = cute.max(-1.0, cute.min(1.0, y))

    # Scale
    gC[b, c, h, w] = y * scale

@cute.jit
def fused_post_host(mA: cute.Tensor, mC: cute.Tensor, add_value: float, scale: float):
    batch, out_c, H, W = mA.shape
    total_elems = batch * out_c * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_post_kernel(mA, mC, add_value, scale).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then applies fused Mish activation, add, Hardtanh, and scaling in a single CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        batch, out_c, H, W = x.shape
        x = x.contiguous().cuda()
        C = torch.empty_like(x)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_host, mA, mC, self.add_value, self.scale)
            self.compiled[key] = compiled

        compiled(mA, mC, self.add_value, self.scale)
        return C