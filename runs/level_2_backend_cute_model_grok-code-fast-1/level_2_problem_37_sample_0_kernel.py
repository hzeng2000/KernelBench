import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def swish_bias_kernel(gX: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    M, N = gX.shape
    ni = thread_idx % N  
    mi = thread_idx // N  

    if mi < M and ni < N:
        x_val = gX[mi, ni]
        bias_val = gBias[ni]
        sig = 1.0 / (1.0 + cute.exp(-x_val))
        gOut[mi, ni] = sig * x_val + bias_val

@cute.jit
def swish_bias_host(mX: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor):
    M = mX.shape[0]
    N = mX.shape[1]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    swish_bias_kernel(mX, mBias, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation fused with bias addition, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.compiled = {}

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
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
            compiled = cute.compile(swish_bias_host, mX, mBias, mOut)
            self.compiled[key] = compiled

        compiled(mX, mBias, mOut)
        x = self.group_norm(out)
        return x