import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_bn_scale_kernel(gX: cute.Tensor, gY: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor, eps: float, scaling_factor: float): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    N, C, H, W = gX.shape
    total_elems = N * C * H * W
    if thread_idx >= total_elems:
        return

    ni = thread_idx % W
    wi = (thread_idx // W) % H
    hi = (thread_idx // (W * H)) % C
    ci = thread_idx // (W * H * C)

    x_val = gX[ci, hi, wi, ni]
    mean_val = gMean[hi]
    var_val = gVar[hi]
    gamma_val = gGamma[hi]
    beta_val = gBeta[hi]

    y_val = (x_val - mean_val) / cute.sqrt(var_val + eps) * gamma_val + beta_val
    gY[ci, hi, wi, ni] = y_val * scaling_factor

@cute.jit
def fused_bn_scale_host(mX: cute.Tensor, mY: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor, eps: float, scaling_factor: float):
    N, C, H, W = mX.shape
    total_elems = N * C * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_bn_scale_kernel(mX, mY, mMean, mVar, mGamma, mBeta, eps, scaling_factor).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model that performs convolution with PyTorch, then fuses Batch Normalization and scaling into a single custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()  # Ensure eval mode for inference
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        x = x.contiguous().cuda()
        y = torch.empty_like(x)

        mean = self.bn.running_mean.contiguous().cuda()
        var = self.bn.running_var.contiguous().cuda()
        gamma = self.bn.weight.contiguous().cuda()
        beta = self.bn.bias.contiguous().cuda()
        eps = self.bn.eps

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGamma = from_dlpack(gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_bn_scale_host, mX, mY, mMean, mVar, mGamma, mBeta, eps, self.scaling_factor)
            self.compiled[key] = compiled

        compiled(mX, mY, mMean, mVar, mGamma, mBeta, eps, self.scaling_factor)
        return y