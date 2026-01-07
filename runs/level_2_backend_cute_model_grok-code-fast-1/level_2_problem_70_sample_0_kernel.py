import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_kernel(gX: cute.Tensor, gOut: cute.Tensor, scaling_factor: float):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    m, n = gX.shape
    total = m * n

    if thread_idx >= total:
        return

    mi = thread_idx // n
    ni = thread_idx % n

    val = gX[mi, ni]
    sig = 1.0 / (1.0 + cute.exp(-val))
    gOut[mi, ni] = sig * scaling_factor + val

@cute.jit
def fused_host(mX: cute.Tensor, mOut: cute.Tensor, scaling_factor: float):
    M, N = mX.shape

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_kernel(mX, mOut, scaling_factor).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd" with fused element-wise operations.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x):
        """
        Forward pass of the optimized model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = self.gemm(x)
        M, N = x.shape
        x = x.contiguous().cuda()
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_host, mX, mOut, self.scaling_factor)
            self.compiled[key] = compiled

        compiled(mX, mOut, self.scaling_factor)
        return out