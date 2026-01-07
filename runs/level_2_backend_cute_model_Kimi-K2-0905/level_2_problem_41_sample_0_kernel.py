import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_bn_gelu_relu_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    M: int, N: int, K: int
):
    # Shared memory for tile
    shared_mem = cute.shared_memory(128 * 128 * 4 + 128 * 128 * 4)  # A tile + B tile
    
    # Thread identifiers
    tidx = cute.thread_idx().x
    tidy = cute.thread_idx().y
    bidx = cute.block_idx().x
    bidy = cute.block_idx().y
    
    # Tile sizes
    TILE_M = 128
    TILE_N = 128
    TILE_K = 8
    
    # Global memory indices
    row = bidx * TILE_M + tidx
    col = bidy * TILE_N + tidy
    
    # Accumulator
    acc = 0.0
    
    # GEMM computation
    for k in range(0, K, TILE_K):
        if row < M and (k + tidy) < K:
            shared_mem[tidx * TILE_K + tidy] = gA[row, k + tidy]
        if col < N and (k + tidx) < K:
            shared_mem[TILE_M * TILE_K + tidy * TILE_K + tidx] = gB[k + tidx, col]
        cute.sync_threads()
        
        # Compute partial dot product
        for i in range(TILE_K):
            if (k + i) < K:
                acc += shared_mem[tidx * TILE_K + i] * shared_mem[TILE_M * TILE_K + i * TILE_K + tidy]
        cute.sync_threads()
    
    # Apply BatchNorm
    if row < M and col < N:
        mean = gMean[col]
        var = gVar[col]
        gamma = gGamma[col]
        beta = gBeta[col]
        
        # Normalize
        x_norm = (acc - mean) / cute.sqrt(var + 1e-5)
        # Scale and shift
        bn_out = gamma * x_norm + beta
        
        # GELU activation
        gelu_out = 0.5 * bn_out * (1.0 + cute.tanh(0.7978845608 * (bn_out + 0.044715 * bn_out * bn_out * bn_out)))
        
        # ReLU activation
        relu_out = cute.max(0.0, gelu_out)
        
        gC[row, col] = relu_out

@cute.jit
def gemm_bn_gelu_relu_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    M: int, N: int, K: int
):
    grid_x = cute.ceil_div(M, 128)
    grid_y = cute.ceil_div(N, 128)
    
    gemm_bn_gelu_relu_kernel(
        mA, mB, mC, mMean, mVar, mGamma, mBeta, M, N, K
    ).launch(grid=(grid_x, grid_y, 1), block=(128, 128, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights for GEMM (transposed for our kernel)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2.0 / in_features))
        
        # BatchNorm parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bn_running_mean', torch.zeros(out_features))
        self.register_buffer('bn_running_var', torch.ones(out_features))
        
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # BatchNorm parameters
        mMean = from_dlpack(self.bn_running_mean, assumed_align=16)
        mVar = from_dlpack(self.bn_running_var, assumed_align=16)
        mGamma = from_dlpack(self.bn_weight, assumed_align=16)
        mBeta = from_dlpack(self.bn_bias, assumed_align=16)
        
        # Compile kernel if not already compiled
        key = (x.dtype, self.in_features, self.out_features)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                gemm_bn_gelu_relu_host,
                mA, mB, mC, mMean, mVar, mGamma, mBeta,
                batch_size, self.out_features, self.in_features
            )
            self.compiled[key] = compiled
        
        # Launch kernel
        compiled(mA, mB, mC, mMean, mVar, mGamma, mBeta,
                 batch_size, self.out_features, self.in_features)
        
        return output