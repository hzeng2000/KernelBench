import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_matmul_bn_bias_div_swish_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gRunningMean: cute.Tensor, gRunningVar: cute.Tensor,
    gGamma: cute.Tensor, gBeta: cute.Tensor,
    gOut: cute.Tensor,
    bias: float, divide_value: float, eps: float,
    M: int, N: int, K: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    
    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy
    
    if row < M and col < N:
        # Matrix multiplication
        acc = 0.0
        for k in range(K):
            acc += gX[row, k] * gW[col, k]
        acc += gB[col]
        
        # Batch normalization (inference mode)
        mean = gRunningMean[col]
        var = gRunningVar[col]
        bn_out = (acc - mean) / cute.sqrt(var + eps)
        bn_out = bn_out * gGamma[col] + gBeta[col]
        
        # Bias addition, division, Swish
        bn_out = bn_out + bias
        bn_out = bn_out / divide_value
        sigmoid = 1.0 / (1.0 + cute.exp(-bn_out))
        swish_out = bn_out * sigmoid
        
        gOut[row, col] = swish_out

@cute.jit
def fused_matmul_bn_bias_div_swish_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mRunningMean: cute.Tensor, mRunningVar: cute.Tensor,
    mGamma: cute.Tensor, mBeta: cute.Tensor,
    mOut: cute.Tensor,
    bias: float, divide_value: float, eps: float
):
    M, K = mX.shape
    N, _ = mW.shape
    
    threads_per_block = 16
    grid_x = cute.ceil_div(M, threads_per_block)
    grid_y = cute.ceil_div(N, threads_per_block)
    
    fused_matmul_bn_bias_div_swish_kernel(
        mX, mW, mB, mRunningMean, mRunningVar, mGamma, mBeta, mOut,
        bias, divide_value, eps, M, N, K
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block, threads_per_block, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.divide_value = divide_value
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_linear = nn.Parameter(torch.randn(out_features))
        
        # Batch norm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Additional bias
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_linear, -bound, bound)
        
        self.compiled = {}
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Prepare output tensor
        out = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.bias_linear, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningMean = from_dlpack(self.running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningVar = from_dlpack(self.running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGamma = from_dlpack(self.bn_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(self.bn_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile key
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_matmul_bn_bias_div_swish_host,
                mX, mW, mB, mRunningMean, mRunningVar, mGamma, mBeta, mOut,
                self.bias.item(), self.divide_value, self.bn_eps
            )
            self.compiled[key] = compiled
        
        # Launch kernel
        compiled(mX, mW, mB, mRunningMean, mRunningVar, mGamma, mBeta, mOut,
                 self.bias.item(), self.divide_value, self.bn_eps)
        
        return out