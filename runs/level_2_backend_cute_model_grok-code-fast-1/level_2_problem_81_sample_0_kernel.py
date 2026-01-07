import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def activation_kernel(gX: cute.Tensor, gY: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    m, n = gX.shape
    ni = thread_idx % n  
    mi = thread_idx // n  

    if mi < m and ni < n:
        x_val = gX[mi, ni]
        # swish: x * sigmoid(x)
        sig = 1.0 / (1.0 + cute.math.exp(-x_val))
        swish = x_val * sig
        # / 2.0
        div = swish / 2.0
        # clamp -1 to 1
        clamped1 = cute.math.max(cute.math.min(div, 1.0), -1.0)
        # tanh
        tanh_val = cute.math.tanh(clamped1)
        # clamp -1 to 1
        clamped2 = cute.math.max(cute.math.min(tanh_val, 1.0), -1.0)
        gY[mi, ni] = clamped2

@cute.jit
def activation_host(mX: cute.Tensor, mY: cute.Tensor):
    M = mX.shape[0]
    N = mX.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    activation_kernel(mX, mY).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.compiled = {}

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        M, N = x.shape
        x = x.contiguous().cuda()
        y = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(activation_host, mX, mY)
            self.compiled[key] = compiled

        compiled(mX, mY)
        return y