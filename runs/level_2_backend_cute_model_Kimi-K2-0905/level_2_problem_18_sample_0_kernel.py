import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_linear_sum_max_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gb: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, in_features: int, out_features: int
):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    bdim = cute.arch.block_dim().x

    row = bidx
    if row < batch_size:
        max_val = -1e38
        sum_val = 0.0
        
        for col in range(out_features):
            idx = row * out_features + col
            accum = gb[col]
            for k in range(in_features):
                accum += gX[row, k] * gW[col, k]
            
            if accum > max_val:
                max_val = accum
            sum_val += accum
            
        gOut[row, 0] = max_val

@cute.jit
def fused_linear_sum_max_host(
    mX: cute.Tensor, mW: cute.Tensor, mb: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, in_features: int, out_features: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(batch_size, 1)
    
    fused_linear_sum_max_kernel(
        mX, mW, mb, mOut, batch_size, in_features, out_features
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def logsumexp_kernel(gX: cute.Tensor, gOut: cute.Tensor, batch_size: int):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    
    if bidx < batch_size and tidx == 0:
        x = gX[bidx, 0]
        gOut[bidx, 0] = math.log1p(math.exp(x))

@cute.jit
def logsumexp_host(mX: cute.Tensor, mOut: cute.Tensor, batch_size: int):
    logsumexp_kernel(mX, mOut, batch_size).launch(
        grid=(batch_size, 1, 1), block=(1, 1, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        temp1 = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
        temp2 = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mb = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mTemp1 = from_dlpack(temp1, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mTemp2 = from_dlpack(temp2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        key = (x.dtype, batch_size, self.in_features, self.out_features)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_linear_sum_max_host, mX, mW, mb, mTemp1, batch_size, self.in_features, self.out_features)
            self.compiled[key] = compiled
            
        compiled(mX, mW, mb, mTemp1, batch_size, self.in_features, self.out_features)
        
        mean_val = temp1.mean()
        temp1.fill_(mean_val)
        
        key2 = (x.dtype, batch_size)
        compiled2 = self.compiled.get(key2)
        if compiled2 is None:
            compiled2 = cute.compile(logsumexp_host, mTemp1, mTemp2, batch_size)
            self.compiled[key2] = compiled2
            
        compiled2(mTemp1, mTemp2, batch_size)
        compiled2(mTemp2, mTemp1, batch_size)
        
        return temp1