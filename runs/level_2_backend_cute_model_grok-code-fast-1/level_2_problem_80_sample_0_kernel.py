import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def max_kernel(gX: cute.Tensor, gMaxes: cute.Tensor):
    bidx = cute.arch.block_idx().x
    tidx = cute.arch.thread_idx().x
    bdim = cute.arch.block_dim().x
    j = bidx
    i = tidx
    M, N = gX.shape
    if j < N and i < M:
        shared = cute.shared_memory(float, (bdim,))
        shared[tidx] = gX[i, j]
        cute.sync()
        # Simple reduction for max
        step = bdim // 2
        while step > 0:
            if tidx < step:
                shared[tidx] = max(shared[tidx], shared[tidx + step])
            cute.sync()
            step //= 2
        if tidx == 0:
            gMaxes[0, j] = shared[0]

@cute.kernel
def gelu_kernel(gMaxes: cute.Tensor, gMean: cute.Tensor, gY: cute.Tensor):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x
    j = bidx * bdim + tidx
    N = gMaxes.shape[1]
    if j < N:
        mean_val = gMean[0]
        val = gMaxes[0, j] - mean_val
        # GELU approximation
        gY[0, j] = 0.5 * val * (1.0 + cute.tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.max_dim = max_dim
        self.compiled_max = None
        self.compiled_gelu = None

    def forward(self, x):
        batch_size, _ = x.shape
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        C = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)

        # GEMM with CUTLASS
        gemm_op = cutlass.ops.Gemm(element_accumulator_dtype=torch.float32, element_output_dtype=torch.float32)
        gemm_op.run(x, weight.t(), C, bias=bias)

        # Max per column
        maxes = torch.empty(1, out_features, dtype=C.dtype, device=C.device)
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mMaxes = from_dlpack(maxes, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        if self.compiled_max is None:
            self.compiled_max = cute.compile(max_kernel, mC, mMaxes)
        max_kernel(mC, mMaxes).launch(grid=(out_features, 1, 1), block=(batch_size, 1, 1))

        # Compute mean
        mean_val = maxes.mean(dim=1, keepdim=True)
        mean_tensor = mean_val.contiguous()

        # GELU
        y = torch.empty(1, out_features, dtype=C.dtype, device=C.device)
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mMean = from_dlpack(mean_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        if self.compiled_gelu is None:
            self.compiled_gelu = cute.compile(gelu_kernel, mMaxes, mMean, mY)
        gelu_kernel(mMaxes, mMean, mY).launch(grid=(cute.ceil_div(out_features, 256), 1, 1), block=(256, 1, 1))

        return y