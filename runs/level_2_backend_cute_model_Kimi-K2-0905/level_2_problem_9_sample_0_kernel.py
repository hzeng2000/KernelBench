import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_bias_sub_mul_relu_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, gD: cute.Tensor,
    subtract_val: float, multiply_val: float
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    thread_idx_x = bidx * bdimx + tidx
    thread_idx_y = bidy * bdimy + tidy
    
    m, k = gA.shape
    n, _ = gB.shape
    
    if thread_idx_x < m and thread_idx_y < n:
        acc = 0.0
        for ki in range(k):
            acc += gA[thread_idx_x, ki] * gB[thread_idx_y, ki]
        
        acc = acc + gC[thread_idx_y]
        acc = acc - subtract_val
        acc = acc * multiply_val
        acc = max(acc, 0.0)
        
        gD[thread_idx_x, thread_idx_y] = acc

@cute.jit
def fused_gemm_bias_sub_mul_relu_host(
    mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor,
    subtract_val: float, multiply_val: float
):
    M, K = mA.shape
    N, _ = mB.shape
    
    threads_per_block_x = 16
    threads_per_block_y = 16
    grid_x = cute.ceil_div(M, threads_per_block_x)
    grid_y = cute.ceil_div(N, threads_per_block_y)
    
    fused_gemm_bias_sub_mul_relu_kernel(
        mA, mB, mBias, mOut, subtract_val, multiply_val
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block_x, threads_per_block_y, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        weight = self.linear.weight.data.t().contiguous()
        bias = self.linear.bias.data.contiguous()
        
        output = torch.empty((batch_size, self.linear.out_features), dtype=x.dtype, device=x.device)
        
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_gemm_bias_sub_mul_relu_host, 
                mA, mB, mBias, mOut, 
                self.subtract_value, self.multiply_value
            )
            self.compiled[key] = compiled
        
        compiled(mA, mB, mBias, mOut)
        return output