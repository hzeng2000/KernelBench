import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_logsumexp_activations_kernel(gX: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor):
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bdim = cute.arch.block_dim()[0]
    batch_idx = bidx * bdim + tidx
    if batch_idx >= gX.shape[0]:
        return
    out_features = gX.shape[1]
    max_val = float('-inf')
    for j in range(out_features):
        val = gX[batch_idx, j] + gBias[j]
        if val > max_val:
            max_val = val
    sum_exp = 0.0
    for j in range(out_features):
        val = gX[batch_idx, j] + gBias[j]
        sum_exp += cute.math.exp(val - max_val)
    lse = max_val + cute.math.log(sum_exp)
    # LeakyReLU
    x = cute.math.max(lse, 0.01 * lse)
    # LeakyReLU again
    x = cute.math.max(x, 0.01 * x)
    # GELU
    sqrt_2_pi = cute.math.sqrt(2.0 / 3.141592653589793)
    coeff = 0.044715
    tanh_arg = sqrt_2_pi * (x + coeff * x * x * x)
    x = 0.5 * x * (1.0 + cute.math.tanh(tanh_arg))
    # GELU again
    tanh_arg = sqrt_2_pi * (x + coeff * x * x * x)
    x = 0.5 * x * (1.0 + cute.math.tanh(tanh_arg))
    gOut[batch_idx, 0] = x

@cute.jit
def fused_logsumexp_activations_host(mX: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor):
    batch_size = mX.shape[0]
    threads_per_block = 256
    grid_x = cute.ceil_div(batch_size, threads_per_block)
    fused_logsumexp_activations_kernel(mX, mBias, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        out_features = self.linear.out_features
        # Perform GEMM using CUTLASS
        A = x.contiguous()
        B = self.linear.weight.t().contiguous()
        C = torch.zeros(batch_size, out_features, dtype=x.dtype, device=x.device)
        cutlass.gemm(A, B, C, alpha=1.0, beta=0.0)
        # Fused kernel for bias add, LogSumExp, and activations
        bias = self.linear.bias
        out = torch.empty(batch_size, 1, dtype=x.dtype, device=x.device)
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_logsumexp_activations_host, mC, mBias, mOut)
            self.compiled[key] = compiled
        compiled(mC, mBias, mOut)
        return out