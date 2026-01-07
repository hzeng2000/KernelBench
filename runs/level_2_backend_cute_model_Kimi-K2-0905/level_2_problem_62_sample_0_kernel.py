import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_matmul_gn_leakyrelu_add_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gOut: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor,
    gGamma: cute.Tensor, gBeta: cute.Tensor,
    batch_size: int, input_size: int, hidden_size: int,
    num_groups: int, eps: float, negative_slope: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy
    
    if row < batch_size and col < hidden_size:
        # Compute matmul
        sum_val = 0.0
        for k in range(input_size):
            sum_val += gX[row, k] * gW[col, k]
        sum_val += gB[col]
        
        # GroupNorm computation
        group_size = hidden_size // num_groups
        group_idx = col // group_size
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        
        # Compute mean for this group
        if col == start_idx:
            group_sum = 0.0
            for i in range(start_idx, end_idx):
                group_sum += sum_val
            gMean[row, group_idx] = group_sum / group_size
        
        cute.arch.barrier_sync()
        
        mean = gMean[row, group_idx]
        
        # Compute variance for this group
        if col == start_idx:
            group_var_sum = 0.0
            for i in range(start_idx, end_idx):
                diff = sum_val - mean
                group_var_sum += diff * diff
            gVar[row, group_idx] = group_var_sum / group_size
        
        cute.arch.barrier_sync()
        
        var = gVar[row, group_idx]
        std = cute.math.sqrt(var + eps)
        
        # Normalize
        normalized = (sum_val - mean) / std
        
        # Apply gamma and beta
        normalized = normalized * gGamma[col] + gBeta[col]
        
        # Leaky ReLU
        if normalized < 0.0:
            normalized = normalized * negative_slope
        
        # Element-wise add (x + x)
        gOut[row, col] = normalized + normalized

@cute.jit
def fused_matmul_gn_leakyrelu_add_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mOut: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor,
    mGamma: cute.Tensor, mBeta: cute.Tensor,
    batch_size: int, input_size: int, hidden_size: int,
    num_groups: int, eps: float, negative_slope: float
):
    threads_per_block = 16
    grid_x = cute.ceil_div(batch_size, threads_per_block)
    grid_y = cute.ceil_div(hidden_size, threads_per_block)
    
    fused_matmul_gn_leakyrelu_add_kernel(
        mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta,
        batch_size, input_size, hidden_size,
        num_groups, eps, negative_slope
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block, threads_per_block, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        
        self.compiled = {}
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        out = torch.empty((batch_size, self.hidden_size), dtype=x.dtype, device=x.device)
        mean = torch.empty((batch_size, self.num_groups), dtype=x.dtype, device=x.device)
        var = torch.empty((batch_size, self.num_groups), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGamma = from_dlpack(self.gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(self.beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_matmul_gn_leakyrelu_add_host,
                mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta,
                batch_size, self.input_size, self.hidden_size,
                self.num_groups, self.eps, self.negative_slope
            )
            self.compiled[key] = compiled
        
        compiled(
            mX, mW, mB, mOut, mMean, mVar, mGamma, mBeta,
            batch_size, self.input_size, self.hidden_size,
            self.num_groups, self.eps, self.negative_slope
        )
        
        return out