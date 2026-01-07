import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_norm_div_kernel(gX: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, eps: float, divide_by: float, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    batch, channels, height, width = gX.shape
    total = batch * channels * height * width

    if thread_idx < total:
        b = thread_idx // (channels * height * width)
        c = (thread_idx // (height * width)) % channels
        h = (thread_idx // width) % height
        w = thread_idx % width

        mean_val = gMean[b, c, 0, 0]
        var_val = gVar[b, c, 0, 0]
        weight_val = gWeight[0, c, 0, 0]
        bias_val = gBias[0, c, 0, 0]
        x_val = gX[b, c, h, w]

        normalized = (x_val - mean_val) / cute.sqrt(var_val + eps)
        gC[b, c, h, w] = (normalized * weight_val + bias_val) / divide_by

@cute.jit
def fused_norm_div_host(mX: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, eps: float, divide_by: float, mC: cute.Tensor):
    batch, channels, height, width = mX.shape
    total_elems = batch * channels * height * width

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_norm_div_kernel(mX, mMean, mVar, mWeight, mBias, eps, divide_by, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies Instance Normalization, and divides by a constant.
    The Instance Normalization and division are fused into a single custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        with torch.no_grad():
            mean = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        weight = self.instance_norm.weight.view(1, -1, 1, 1)
        bias = self.instance_norm.bias.view(1, -1, 1, 1)
        eps = self.instance_norm.eps

        batch, channels, height, width = x.shape
        x = x.contiguous().cuda()
        mean = mean.contiguous().cuda()
        var = var.contiguous().cuda()
        weight = weight.contiguous().cuda()
        bias = bias.contiguous().cuda()
        C = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_norm_div_host, mX, mMean, mVar, mWeight, mBias, eps, self.divide_by, mC)
            self.compiled[key] = compiled

        compiled(mX, mMean, mVar, mWeight, mBias, eps, self.divide_by, mC)
        return C