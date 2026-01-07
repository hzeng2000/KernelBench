import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_scale_bn_kernel(gA: cute.Tensor, gW: cute.Tensor, gScale: cute.Tensor, gRunningMean: cute.Tensor, gRunningVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor, eps: float, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    M, K = gA.shape
    N = gW.shape[0]
    total_elems = M * N

    if thread_idx >= total_elems:
        return

    mi = thread_idx // N  
    ni = thread_idx % N  

    sum_val = 0.0
    for ki in range(K):
        sum_val += gA[mi, ki] * gW[ni, ki]

    c_val = sum_val * gScale[ni]
    c_val = (c_val - gRunningMean[ni]) / cute.sqrt(gRunningVar[ni] + eps) * gGamma[ni] + gBeta[ni]

    gC[mi, ni] = c_val

@cute.jit
def fused_gemm_scale_bn_host(mA: cute.Tensor, mW: cute.Tensor, mScale: cute.Tensor, mRunningMean: cute.Tensor, mRunningVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor, eps: float, mC: cute.Tensor):
    M = mA.shape[0]
    N = mW.shape[0]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_gemm_scale_bn_kernel(mA, mW, mScale, mRunningMean, mRunningVar, mGamma, mBeta, eps, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.compiled = {}

    def forward(self, x):
        M, K = x.shape
        N = self.gemm.out_features
        x = x.contiguous().cuda()
        weight = self.gemm.weight.contiguous().cuda()
        scale = self.scale.contiguous().cuda()
        running_mean = self.bn.running_mean.contiguous().cuda()
        running_var = self.bn.running_var.contiguous().cuda()
        gamma = self.bn.weight.contiguous().cuda()
        beta = self.bn.bias.contiguous().cuda()
        eps = self.bn.eps
        C = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mScale = from_dlpack(scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mRunningMean = from_dlpack(running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mRunningVar = from_dlpack(running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGamma = from_dlpack(gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBeta = from_dlpack(beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_scale_bn_host, mA, mW, mScale, mRunningMean, mRunningVar, mGamma, mBeta, eps, mC)
            self.compiled[key] = compiled

        compiled(mA, mW, mScale, mRunningMean, mRunningVar, mGamma, mBeta, eps, mC)
        return C