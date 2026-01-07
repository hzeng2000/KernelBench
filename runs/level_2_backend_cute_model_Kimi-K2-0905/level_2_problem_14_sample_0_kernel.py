import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_div_sum_scale_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
                              batch_size: int, hidden_size: int, input_size: int,
                              div_factor: float, scale_factor: float):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    thread_idx_x = bidx * bdimx + tidx
    thread_idx_y = bidy * bdimy + tidy
    
    if thread_idx_y < batch_size and thread_idx_x < 1:
        sum_val = 0.0
        for k in range(input_size):
            a_val = gA[thread_idx_y, k]
            b_val = gB[0, k]  # weight row
            sum_val += a_val * b_val
        
        sum_val = sum_val / div_factor
        sum_val = sum_val * scale_factor
        gC[thread_idx_y, 0] = sum_val

@cute.jit
def gemm_div_sum_scale_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                            batch_size: int, hidden_size: int, input_size: int,
                            div_factor: float, scale_factor: float):
    threads_per_block = 256
    grid_x = cute.ceil_div(1, threads_per_block)
    grid_y = cute.ceil_div(batch_size, threads_per_block)
    
    gemm_div_sum_scale_kernel(mA, mB, mC, batch_size, hidden_size, input_size,
                             div_factor, scale_factor).launch(
        grid=(grid_x, grid_y, 1), 
        block=(threads_per_block, threads_per_block, 1)
    )

@cute.kernel
def gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
                M: int, N: int, K: int):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy
    
    if row < M and col < N:
        sum_val = 0.0
        for k in range(K):
            sum_val += gA[row, k] * gB[col, k]
        gC[row, col] = sum_val / 2.0

@cute.jit
def gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
              M: int, N: int, K: int):
    threads_per_block = 16
    grid_x = cute.ceil_div(M, threads_per_block)
    grid_y = cute.ceil_div(N, threads_per_block)
    
    gemm_kernel(mA, mB, mC, M, N, K).launch(
        grid=(grid_x, grid_y, 1),
        block=(threads_per_block, threads_per_block, 1)
    )

@cute.kernel
def reduce_sum_kernel(gA: cute.Tensor, gC: cute.Tensor,
                      batch_size: int, hidden_size: int, scale_factor: float):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    
    row = bidx
    if row < batch_size:
        sum_val = 0.0
        for i in range(hidden_size):
            sum_val += gA[row, i]
        
        gC[row, 0] = sum_val * scale_factor

@cute.jit
def reduce_sum_host(mA: cute.Tensor, mC: cute.Tensor,
                    batch_size: int, hidden_size: int, scale_factor: float):
    threads_per_block = 256
    grid_x = batch_size
    
    reduce_sum_kernel(mA, mC, batch_size, hidden_size, scale_factor).launch(
        grid=(grid_x, 1, 1),
        block=(threads_per_block, 1, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.contiguous().cuda()
        weight_T = self.weight.T.contiguous().cuda()
        
        # Gemm + Division fusion
        C1 = torch.empty((batch_size, self.hidden_size), dtype=x.dtype, device=x.device)
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight_T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC1 = from_dlpack(C1, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key1 = (x.dtype, "gemm")
        compiled1 = self.compiled.get(key1)
        if compiled1 is None:
            compiled1 = cute.compile(gemm_host, mA, mB, mC1, batch_size, self.hidden_size, self.input_size)
            self.compiled[key1] = compiled1
        
        compiled1(mA, mB, mC1, batch_size, self.hidden_size, self.input_size)
        
        # Reduction sum + scaling fusion
        C2 = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
        mC2 = from_dlpack(C2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key2 = (x.dtype, "reduce")
        compiled2 = self.compiled.get(key2)
        if compiled2 is None:
            compiled2 = cute.compile(reduce_sum_host, mC1, mC2, batch_size, self.hidden_size, self.scaling_factor)
            self.compiled[key2] = compiled2
        
        compiled2(mC1, mC2, batch_size, self.hidden_size, self.scaling_factor)
        
        return C2