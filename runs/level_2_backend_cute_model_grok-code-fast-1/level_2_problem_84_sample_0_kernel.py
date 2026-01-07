import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_bn_scale_kernel(gA: cute.Tensor, gB: cute.Tensor, gBias: cute.Tensor, gBnWeight: cute.Tensor, gBnBias: cute.Tensor, gRunningMean: cute.Tensor, gRunningVar: cute.Tensor, eps: float, scale: float, gC: cute.Tensor):
    M, K = gA.shape
    N = gB.shape[0]
    
    tidx = cute.arch.thread_idx(0)
    tidy = cute.arch.thread_idx(1)
    bidx = cute.arch.block_idx(0)
    bidy = cute.arch.block_idx(1)
    bdimx = cute.arch.block_dim(0)
    bdimy = cute.arch.block_dim(1)
    
    i = bidx * bdimx + tidx
    j = bidy * bdimy + tidy
    
    if i < M and j < N:
        sum_val = 0.0
        for k in range(K):
            sum_val += gA[i, k] * gB[j, k]
        sum_val += gBias[j]
        sum_val = gBnWeight[j] * (sum_val - gRunningMean[j]) / cute.sqrt(gRunningVar[j] + eps) + gBnBias[j]
        gC[i, j] = sum_val * scale

@cute.jit
def fused_gemm_bn_scale_host(mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor, mBnWeight: cute.Tensor, mBnBias: cute.Tensor, mRunningMean: cute.Tensor, mRunningVar: cute.Tensor, eps: float, scale: float, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    
    block_dim = (32, 32, 1)
    grid = (cute.ceil_div(M, block_dim[0]), cute.ceil_div(N, block_dim[1]), 1)
    
    fused_gemm_bn_scale_kernel(mA, mB, mBias, mBnWeight, mBnBias, mRunningMean, mRunningVar, eps, scale, mC).launch(grid=grid, block=block_dim)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)
        self.compiled = {}

    def forward(self, x):
        M, K = x.shape
        N = self.gemm.out_features
        
        x = x.contiguous().cuda()
        W = self.gemm.weight.t().contiguous().cuda()  # W^T for GEMM
        bias = self.gemm.bias.contiguous().cuda()
        bn_weight = self.bn.weight.contiguous().cuda()
        bn_bias = self.bn.bias.contiguous().cuda()
        running_mean = self.bn.running_mean.contiguous().cuda()
        running_var = self.bn.running_var.contiguous().cuda()
        eps = self.bn.eps
        scale = self.scale.item()
        
        C = torch.empty((M, N), dtype=x.dtype, device=x.device)
        
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBnWeight = from_dlpack(bn_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBnBias = from_dlpack(bn_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningMean = from_dlpack(running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningVar = from_dlpack(running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_bn_scale_host, mA, mB, mBias, mBnWeight, mBnBias, mRunningMean, mRunningVar, eps, scale, mC)
            self.compiled[key] = compiled
        
        compiled(mA, mB, mBias, mBnWeight, mBnBias, mRunningMean, mRunningVar, eps, scale, mC)
        
        return self.softmax(C)