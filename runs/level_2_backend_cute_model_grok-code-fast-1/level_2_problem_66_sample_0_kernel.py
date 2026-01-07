import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gBias: cute.Tensor, gC: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    M, K = gA.shape
    N, _ = gB.shape
    total_elems = M * N

    if thread_idx >= total_elems:
        return

    i = thread_idx // N
    j = thread_idx % N

    sum_val = 0.0
    for k in range(K):
        sum_val += gA[i, k] * gB[j, k]

    gC[i, j] = sum_val + gBias[j]

@cute.jit
def gemm_host(mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]

    threads_per_block = 256
    total_elems = M * N
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    gemm_kernel(mA, mB, mBias, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication with custom CuTe kernel, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.compiled = {}

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        A = x.contiguous().cuda()
        B = self.matmul.weight.contiguous().cuda()
        bias = self.matmul.bias.contiguous().cuda()
        M, K = A.shape
        N, _ = B.shape
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (A.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(gemm_host, mA, mB, mBias, mC)
            self.compiled[key] = compiled

        compiled(mA, mB, mBias, mC)
        x = self.dropout(C)
        x = torch.softmax(x, dim=1)
        return x