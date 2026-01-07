import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_post_kernel(gX: cute.Tensor, gY: cute.Tensor, gScaleFactor: cute.Tensor, gRunningMean: cute.Tensor, gRunningVar: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gEps: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_id = bidx * bdim + tidx

    batch, channel, depth, height, width = gX.shape
    num_bc = batch * channel
    if thread_id >= num_bc:
        return

    b = thread_id // channel
    c = thread_id % channel

    sum_val = 0.0
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                sum_val += gX[b, c, d, h, w]
    avg = sum_val / (depth * height * width)

    scale_factor = gScaleFactor[0]
    eps = gEps[0]
    std = cute.sqrt(gRunningVar[c] + eps)
    gY[b, c, 0, 0, 0] = (gWeight[c] / std) * (scale_factor * avg) - (gWeight[c] / std) * gRunningMean[c] + gBias[c]

@cute.jit
def fused_post_host(mX: cute.Tensor, mY: cute.Tensor, mScaleFactor: cute.Tensor, mRunningMean: cute.Tensor, mRunningVar: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mEps: cute.Tensor):
    batch, channel, depth, height, width = mX.shape
    num_threads = batch * channel
    threads_per_block = 256
    grid_x = cute.ceil_div(num_threads, threads_per_block)
    fused_post_kernel(mX, mY, mScaleFactor, mRunningMean, mRunningVar, mWeight, mBias, mEps).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, then fuses scaling, batch normalization, 
    and global average pooling into a single custom CuTe (CUTLASS) kernel for speedup.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        M, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        y = torch.empty((M, C, 1, 1, 1), dtype=x.dtype, device=x.device)

        running_mean = self.batch_norm.running_mean.contiguous().cuda()
        running_var = self.batch_norm.running_var.contiguous().cuda()
        weight = self.batch_norm.weight.contiguous().cuda()
        bias = self.batch_norm.bias.contiguous().cuda()
        scale_factor_tensor = torch.tensor([self.scale_factor], dtype=x.dtype, device=x.device)
        eps_tensor = torch.tensor([self.batch_norm.eps], dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mScaleFactor = from_dlpack(scale_factor_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningMean = from_dlpack(running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningVar = from_dlpack(running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mEps = from_dlpack(eps_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_host, mX, mY, mScaleFactor, mRunningMean, mRunningVar, mWeight, mBias, mEps)
            self.compiled[key] = compiled

        compiled(mX, mY, mScaleFactor, mRunningMean, mRunningVar, mWeight, mBias, mEps)
        return y