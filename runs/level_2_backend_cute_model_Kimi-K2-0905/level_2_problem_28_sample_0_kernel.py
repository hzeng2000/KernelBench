import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_linear_instancenorm_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor, gOut: cute.Tensor,
    eps: float, batch_size: int, in_features: int, out_features: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy
    
    if row < batch_size and col < out_features:
        # Linear layer: compute dot product
        sum_val = 0.0
        for k in range(in_features):
            sum_val += gX[row, k] * gW[col, k]
        sum_val += gB[col]
        
        # Instance norm: normalize across batch dimension
        mean = gMean[col]
        var = gVar[col]
        inv_std = cute.rsqrt(var + eps)
        normalized = (sum_val - mean) * inv_std
        
        gOut[row, col] = normalized

@cute.kernel
def compute_stats_kernel(
    gX: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor,
    batch_size: int, out_features: int
):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    
    col = bidx * cute.arch.block_dim().x + tidx
    
    if col < out_features:
        # Compute mean
        sum_val = 0.0
        for row in range(batch_size):
            sum_val += gX[row, col]
        mean = sum_val / batch_size
        gMean[col] = mean
        
        # Compute variance
        var_sum = 0.0
        for row in range(batch_size):
            diff = gX[row, col] - mean
            var_sum += diff * diff
        gVar[col] = var_sum / batch_size

@cute.kernel
def residual_mul_kernel(
    gX: cute.Tensor, gY: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, out_features: int
):
    tidx = cute.arch.thread_idx().x
    bidx = cute.arch.block_idx().x
    
    idx = bidx * cute.arch.block_dim().x + tidx
    
    if idx < batch_size * out_features:
        row = idx // out_features
        col = idx % out_features
        
        temp = gX[row, col] + gY[row, col]
        gOut[row, col] = temp * gY[row, col]

@cute.jit
def fused_linear_instancenorm_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor, mOut: cute.Tensor,
    eps: float, batch_size: int, in_features: int, out_features: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(batch_size, threads_per_block)
    grid_y = cute.ceil_div(out_features, threads_per_block)
    
    fused_linear_instancenorm_kernel(
        mX, mW, mB, mMean, mVar, mOut, eps,
        batch_size, in_features, out_features
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block, threads_per_block, 1))

@cute.jit
def compute_stats_host(
    mX: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor,
    batch_size: int, out_features: int
):
    threads_per_block = 256
    grid_x = cute.ceil_div(out_features, threads_per_block)
    
    compute_stats_kernel(mX, mMean, mVar, batch_size, out_features
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.jit
def residual_mul_host(
    mX: cute.Tensor, mY: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, out_features: int
):
    threads_per_block = 256
    total_elems = batch_size * out_features
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    
    residual_mul_kernel(mX, mY, mOut, batch_size, out_features
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.running_mean = torch.zeros(out_features).cuda()
        self.running_var = torch.ones(out_features).cuda()
        
        self.compiled = {}

    def forward(self, x, y):
        batch_size = x.shape[0]
        
        x = x.contiguous().cuda()
        y = y.contiguous().cuda()
        
        # Allocate intermediate tensors
        linear_out = torch.empty((batch_size, self.out_features), dtype=x.dtype, device=x.device)
        mean = torch.empty((self.out_features,), dtype=x.dtype, device=x.device)
        var = torch.empty((self.out_features,), dtype=x.dtype, device=x.device)
        final_out = torch.empty((batch_size, self.out_features), dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mLinearOut = from_dlpack(linear_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mFinalOut = from_dlpack(final_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile kernels if not already compiled
        key = (x.dtype, batch_size, self.in_features, self.out_features)
        compiled = self.compiled.get(key)
        
        if compiled is None:
            compiled = {
                'linear_norm': cute.compile(fused_linear_instancenorm_host, mX, mW, mB, mMean, mVar, mLinearOut, self.eps, batch_size, self.in_features, self.out_features),
                'compute_stats': cute.compile(compute_stats_host, mLinearOut, mMean, mVar, batch_size, self.out_features),
                'residual_mul': cute.compile(residual_mul_host, mLinearOut, mY, mFinalOut, batch_size, self.out_features)
            }
            self.compiled[key] = compiled
        
        # Compute linear output first
        compiled['linear_norm'](mX, mW, mB, mMean, mVar, mLinearOut, self.eps, batch_size, self.in_features, self.out_features)
        
        # Compute statistics for instance norm
        compiled['compute_stats'](mLinearOut, mMean, mVar, batch_size, self.out_features)
        
        # Apply instance normalization manually
        linear_out = (linear_out - mean.unsqueeze(0)) * torch.rsqrt(var.unsqueeze(0) + self.eps)
        
        # Convert normalized output back to CuTe tensor
        mLinearOut = from_dlpack(linear_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Final residual and multiplication
        compiled['residual_mul'](mLinearOut, mY, mFinalOut, batch_size, self.out_features)
        
        return final_out