import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_scale_bn_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    gScale: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor,
    gGamma: cute.Tensor, gBeta: cute.Tensor, gOut: cute.Tensor,
    eps: float
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    m = gA.shape[0]
    n = gB.shape[0]
    k = gA.shape[1]
    
    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy
    
    if row < m and col < n:
        acc = 0.0
        for ki in range(k):
            acc += gA[row, ki] * gB[col, ki]
        
        acc *= gScale[col]
        
        mean = gMean[col]
        var = gVar[col]
        gamma = gGamma[col]
        beta = gBeta[col]
        
        x_hat = (acc - mean) / cute.sqrt(var + eps)
        gOut[row, col] = gamma * x_hat + beta

@cute.jit
def gemm_scale_bn_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    mScale: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor,
    mGamma: cute.Tensor, mBeta: cute.Tensor, mOut: cute.Tensor,
    eps: float
):
    M = mA.shape[0]
    N = mB.shape[0]
    
    threads_per_block = 16
    grid_x = cute.ceil_div(M, threads_per_block)
    grid_y = cute.ceil_div(N, threads_per_block)
    
    gemm_scale_bn_kernel(
        mA, mB, mC, mScale, mMean, mVar, mGamma, mBeta, mOut, eps
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block, threads_per_block, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(torch.empty(batch_size, self.out_features, device=x.device), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mScale = from_dlpack(self.scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mMean = from_dlpack(self.running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mVar = from_dlpack(self.running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGamma = from_dlpack(torch.ones_like(self.running_mean), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(torch.zeros_like(self.running_mean), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(torch.empty(batch_size, self.out_features, device=x.device), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(gemm_scale_bn_host, mA, mB, mC, mScale, mMean, mVar, mGamma, mBeta, mOut, self.eps)
            self.compiled[key] = compiled
        
        compiled(mA, mB, mC, mScale, mMean, mVar, mGamma, mBeta, mOut)
        
        return mOut