import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_swish_div_clamp_tanh_clamp_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    alpha: float, beta: float, div_val: float
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    
    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy
    
    M = gA.shape[0]
    N = gB.shape[0]
    K = gA.shape[1]
    
    if row < M and col < N:
        acc = 0.0
        for k in range(K):
            acc += gA[row, k] * gB[col, k]
        
        # Swish: x * sigmoid(x)
        sigmoid = 1.0 / (1.0 + cute.exp(-acc))
        swish = acc * sigmoid
        
        # Divide by 2.0
        div_result = swish / div_val
        
        # Clamp between -1.0 and 1.0
        clamp1 = cute.max(cute.min(div_result, 1.0), -1.0)
        
        # Tanh
        tanh_result = cute.tanh(clamp1)
        
        # Final clamp between -1.0 and 1.0
        final_result = cute.max(cute.min(tanh_result, 1.0), -1.0)
        
        gC[row, col] = final_result

@cute.jit
def fused_gemm_activation_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    div_val: float
):
    M = mA.shape[0]
    N = mB.shape[0]
    
    threads_per_block = 16
    grid_x = cute.ceil_div(M, threads_per_block)
    grid_y = cute.ceil_div(N, threads_per_block)
    
    gemm_swish_div_clamp_tanh_clamp_kernel(
        mA, mB, mC, 1.0, 0.0, div_val
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block, threads_per_block, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.compiled = {}
        
        # Initialize weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Prepare tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        output = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile and run kernel
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_activation_host, mA, mB, mC, 2.0)
            self.compiled[key] = compiled
            
        compiled(mA, mB, mC, 2.0)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
            
        return output