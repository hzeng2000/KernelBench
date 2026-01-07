import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_bias_hardtanh_mish_kernel(gX: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    m, n = gX.shape
    ni = thread_idx % n
    mi = thread_idx // n

    if mi < m and ni < n:
        val = gX[mi, ni] + gBias[ni]
        val = cute.max(-1.0, cute.min(1.0, val))
        softplus = cute.log(1.0 + cute.exp(val))
        mish = val * cute.tanh(softplus)
        gOut[mi, ni] = mish

@cute.jit
def fused_bias_hardtanh_mish_host(mX: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor):
    M = mX.shape[0]
    N = mX.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_bias_hardtanh_mish_kernel(mX, mBias, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    A model that performs a GEMM, fused BiasAdd+Hardtanh+Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.compiled = {}

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        # Fused bias + hardtanh + mish
        M, N = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_bias_hardtanh_mish_host, mX, mBias, mOut)
            self.compiled[key] = compiled

        compiled(mX, mBias, mOut)
        x = out
        x = self.groupnorm(x)
        return x