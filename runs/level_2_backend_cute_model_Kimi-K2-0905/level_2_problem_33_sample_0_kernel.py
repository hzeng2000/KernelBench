import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_scale_bias_relu_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    gScale: cute.Tensor, gBias: cute.Tensor,
    M: int, N: int, K: int
):
    # Shared memory for tile
    shared_a = cute.shared_tensor((64, 32), dtype=cute.float32)
    shared_b = cute.shared_tensor((32, 64), dtype=cute.float32)
    
    # Thread identifiers
    tid = cute.thread_idx()
    bid = cute.block_idx()
    
    # Tile indices
    tile_m = bid.x
    tile_n = bid.y
    
    # Global thread index
    thread_idx = tid.x
    
    # Compute global memory indices
    row = tile_m * 64 + thread_idx // 16
    col = tile_n * 64 + thread_idx % 16 * 4
    
    # Initialize accumulator
    acc = cute.zeros((4,), dtype=cute.float32)
    
    # Main loop over K dimension
    for k_tile in range(0, K, 32):
        # Load A tile
        if row < M and k_tile + thread_idx % 32 < K:
            shared_a[thread_idx // 32, thread_idx % 32] = gA[row, k_tile + thread_idx % 32]
        else:
            shared_a[thread_idx // 32, thread_idx % 32] = 0.0
            
        # Load B tile
        if k_tile + thread_idx // 16 < K and col + cute.arange(4) < N:
            for i in range(4):
                shared_b[thread_idx // 16, thread_idx % 16 * 4 + i] = gB[k_tile + thread_idx // 16, col + i]
        else:
            for i in range(4):
                shared_b[thread_idx // 16, thread_idx % 16 * 4 + i] = 0.0
                
        cute.sync_threads()
        
        # Compute partial dot product
        for k in range(32):
            a_val = shared_a[thread_idx // 32, k]
            for i in range(4):
                b_val = shared_b[k, thread_idx % 16 * 4 + i]
                acc[i] += a_val * b_val
                
        cute.sync_threads()
    
    # Apply scale, bias and ReLU
    if row < M and col + cute.arange(4) < N:
        for i in range(4):
            scaled = acc[i] * gScale[col + i]
            biased = scaled + gBias[col + i]
            relu = cute.max(biased, 0.0)
            gC[row, col + i] = relu

@cute.kernel
def fused_bn_kernel(
    gX: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor,
    gGamma: cute.Tensor, gBeta: cute.Tensor, gOut: cute.Tensor,
    eps: float, M: int, N: int
):
    tid = cute.thread_idx()
    bid = cute.block_idx()
    
    row = bid.x * 128 + tid.x
    col = bid.y * 4 + tid.y
    
    if row < M and col < N:
        x_val = gX[row, col]
        mean = gMean[col]
        var = gVar[col]
        gamma = gGamma[col]
        beta = gBeta[col]
        
        # Normalize
        x_norm = (x_val - mean) / cute.sqrt(var + eps)
        
        # Scale and shift
        gOut[row, col] = gamma * x_norm + beta

@cute.jit
def fused_gemm_scale_bias_relu_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    mScale: cute.Tensor, mBias: cute.Tensor
):
    M, K = mA.shape
    K_, N = mB.shape
    
    grid = cute.ceil_div(M, 64), cute.ceil_div(N, 64)
    block = 256, 1
    
    gemm_scale_bias_relu_kernel(
        mA, mB, mC, mScale, mBias,
        M, N, K
    ).launch(grid=grid, block=block)

@cute.jit
def fused_bn_host(
    mX: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor,
    mGamma: cute.Tensor, mBeta: cute.Tensor, mOut: cute.Tensor,
    eps: float
):
    M, N = mX.shape
    
    grid = cute.ceil_div(M, 128), cute.ceil_div(N, 4)
    block = 128, 4
    
    fused_bn_kernel(
        mX, mMean, mVar, mGamma, mBeta, mOut,
        eps, M, N
    ).launch(grid=grid, block=block)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        
        # Batch norm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Ensure contiguous and on CUDA
        x = x.contiguous().cuda()
        
        # Allocate output tensor
        gemm_out = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGemmOut = from_dlpack(gemm_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mScale = from_dlpack(self.scale, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        # Compile and run fused GEMM + scale + bias + ReLU
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_scale_bias_relu_host, mX, mW, mGemmOut, mScale, mBias)
            self.compiled[key] = compiled
        
        compiled(mX, mW, mGemmOut, mScale, mBias)
        
        # Compute batch norm statistics
        if self.training:
            mean = gemm_out.mean(dim=0)
            var = gemm_out.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Allocate output tensor for batch norm
        bn_out = torch.empty_like(gemm_out)
        
        # Convert to CuTe tensors for batch norm
        mGemmOut2 = from_dlpack(gemm_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGamma = from_dlpack(self.bn_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(self.bn_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBnOut = from_dlpack(bn_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile and run fused batch norm
        bn_key = (x.dtype, 'bn')
        bn_compiled = self.compiled.get(bn_key)
        if bn_compiled is None:
            bn_compiled = cute.compile(fused_bn_host, mGemmOut2, mMean, mVar, mGamma, mBeta, mBnOut, self.eps)
            self.compiled[bn_key] = bn_compiled
        
        bn_compiled(mGemmOut2, mMean, mVar, mGamma, mBeta, mBnOut, self.eps)
        
        return bn_out