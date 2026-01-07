import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_kernel(gA: cute.Tensor, gW: cute.Tensor, gBias: cute.Tensor, gBnWeight: cute.Tensor, gBnBias: cute.Tensor, gRunningMean: cute.Tensor, gRunningVar: cute.Tensor, gC: cute.Tensor, eps: float):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    batch_size, in_features = gA.shape
    out_features, _ = gW.shape
    total_elems = batch_size * out_features

    if thread_idx >= total_elems:
        return

    i = thread_idx // out_features
    j = thread_idx % out_features

    # GEMM
    sum_val = 0.0
    for k in range(in_features):
        sum_val += gA[i, k] * gW[j, k]
    sum_val += gBias[j]

    # BatchNorm
    mean = gRunningMean[j]
    var = gRunningVar[j]
    x_hat = (sum_val - mean) / cute.sqrt(var + eps)
    out = x_hat * gBnWeight[j] + gBnBias[j]

    # GELU approximation
    out = 0.5 * out * (1 + cute.tanh(cute.sqrt(2 / 3.141592653589793) * (out + 0.044715 * out * out * out)))

    # ReLU
    out = cute.max(out, 0.0)

    gC[i, j] = out

@cute.jit
def fused_host(mA: cute.Tensor, mW: cute.Tensor, mBias: cute.Tensor, mBnWeight: cute.Tensor, mBnBias: cute.Tensor, mRunningMean: cute.Tensor, mRunningVar: cute.Tensor, mC: cute.Tensor, eps: float):
    batch_size, _ = mA.shape
    out_features, _ = mW.shape
    total_elems = batch_size * out_features
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_kernel(mA, mW, mBias, mBnWeight, mBnBias, mRunningMean, mRunningVar, mC, eps).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.bn_weight = torch.nn.Parameter(torch.ones(out_features))
        self.bn_bias = torch.nn.Parameter(torch.zeros(out_features))
        self.running_mean = torch.zeros(out_features)
        self.running_var = torch.ones(out_features)
        self.eps = 1e-5
        self.compiled = {}

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        batch_size, in_features = A.shape
        out_features = self.weight.shape[0]
        A = A.contiguous().cuda()
        W = self.weight.contiguous().cuda()
        Bias = self.bias.contiguous().cuda()
        BnWeight = self.bn_weight.contiguous().cuda()
        BnBias = self.bn_bias.contiguous().cuda()
        RunningMean = self.running_mean.contiguous().cuda()
        RunningVar = self.running_var.contiguous().cuda()
        C = torch.empty((batch_size, out_features), dtype=A.dtype, device=A.device)

        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(Bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBnWeight = from_dlpack(BnWeight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBnBias = from_dlpack(BnBias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningMean = from_dlpack(RunningMean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningVar = from_dlpack(RunningVar, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (A.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_host, mA, mW, mBias, mBnWeight, mBnBias, mRunningMean, mRunningVar, mC, self.eps)
            self.compiled[key] = compiled

        compiled(mA, mW, mBias, mBnWeight, mBnBias, mRunningMean, mRunningVar, mC, self.eps)
        return C