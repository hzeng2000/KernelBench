import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.ops import conv2d_transpose

@cute.kernel
def post_process_kernel(gX: cute.Tensor, gBias: cute.Tensor, scale: float, gOut: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    batch, out_c, h, w = gX.shape
    total_elems = batch * out_c * h * w

    if thread_idx < total_elems:
        bi = thread_idx // (out_c * h * w)
        ci = (thread_idx % (out_c * h * w)) // (h * w)
        hi = (thread_idx % (h * w)) // w
        wi = thread_idx % w

        val = gX[bi, ci, hi, wi]
        bias_val = gBias[ci, 0, 0]
        val = val + bias_val
        val = cute.max(0.0, cute.min(1.0, val))
        val = val * scale
        val = cute.max(0.0, cute.min(1.0, val))
        val = val / scale
        gOut[bi, ci, hi, wi] = val

@cute.jit
def conv_transpose_host(mA: cute.Tensor, mW: cute.Tensor, mC: cute.Tensor, stride, padding, output_padding):
    conv2d_transpose(mA, mW, mC, stride=stride, padding=padding, output_padding=output_padding)

@cute.jit
def post_process_host(mX: cute.Tensor, mBias: cute.Tensor, scale: float, mOut: cute.Tensor):
    batch, out_c, h, w = mX.shape
    total_elems = batch * out_c * h * w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    post_process_kernel(mX, mBias, scale, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution using CuTe (CUTLASS), adds a bias term, clamps, scales, clamps, and divides using a fused CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self.compiled_conv = {}
        self.compiled_post = {}

    def forward(self, x):
        A = x.contiguous().cuda()
        W = self.conv_transpose.weight.contiguous().cuda()
        batch, in_c, hin, win = A.shape
        out_c = W.shape[1]
        h_out = (hin - 1) * self.conv_transpose.stride[0] - 2 * self.conv_transpose.padding[0] + self.conv_transpose.kernel_size[0] + self.conv_transpose.output_padding[0]
        w_out = (win - 1) * self.conv_transpose.stride[1] - 2 * self.conv_transpose.padding[1] + self.conv_transpose.kernel_size[1] + self.conv_transpose.output_padding[1]
        C = torch.empty((batch, out_c, h_out, w_out), dtype=A.dtype, device=A.device)

        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key_conv = (A.dtype, batch, in_c, hin, win, out_c, h_out, w_out)
        compiled_conv = self.compiled_conv.get(key_conv)
        if compiled_conv is None:
            compiled_conv = cute.compile(conv_transpose_host, mA, mW, mC, self.conv_transpose.stride, self.conv_transpose.padding, self.conv_transpose.output_padding)
            self.compiled_conv[key_conv] = compiled_conv

        compiled_conv(mA, mW, mC, self.conv_transpose.stride, self.conv_transpose.padding, self.conv_transpose.output_padding)

        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mOut = mC

        key_post = (C.dtype, batch, out_c, h_out, w_out)
        compiled_post = self.compiled_post.get(key_post)
        if compiled_post is None:
            compiled_post = cute.compile(post_process_host, mC, mBias, self.scaling_factor, mOut)
            self.compiled_post[key_post] = compiled_post

        compiled_post(mC, mBias, self.scaling_factor, mOut)
        return C