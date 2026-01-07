import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_mish_mish_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int
):
    # Shared memory tiles
    sA = cute.smem_tensor(cute.Tile(cute._16, cute._16), dtype=cute.float32)
    sB = cute.smem_tensor(cute.Tile(cute._16, cute._16), dtype=cute.float32)
    
    # Thread-local accumulator
    acc = cute.zeros(cute.Tile(cute._16, cute._16), dtype=cute.float32)
    
    # Thread indices
    tidx, tidy = cute.arch.thread_idx()[:2]
    bidx, bidy = cute.arch.block_idx()[:2]
    
    # Global position
    row = bidx * cute._16 + tidx
    col = bidy * cute._16 + tidy
    
    # Loop over K dimension
    for k in range(0, K, cute._16):
        # Load tiles to shared memory
        if row < M and k + tidx < K:
            sA[tidx, tidy] = gA[row, k + tidy]
        else:
            sA[tidx, tidy] = 0.0
            
        if col < N and k + tidy < K:
            sB[tidx, tidy] = gB[k + tidx, col]
        else:
            sB[tidx, tidy] = 0.0
        
        cute.arch.sync_threads()
        
        # Compute partial dot product
        for ki in range(cute._16):
            a_val = sA[tidx, ki]
            b_val = sB[ki, tidy]
            acc = acc + a_val * b_val
        
        cute.arch.sync_threads()
    
    # Apply Mish activation twice
    if row < M and col < N:
        val = acc[tidx, tidy]
        
        # First Mish
        tanh_arg = math.tanh(math.log1p(math.exp(val)))
        mish1 = val * tanh_arg
        
        # Second Mish
        tanh_arg2 = math.tanh(math.log1p(math.exp(mish1)))
        mish2 = mish1 * tanh_arg2
        
        gC[row, col] = mish2

@cute.jit
def gemm_mish_mish_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    M: int, N: int, K: int
):
    block_size = 16
    grid_x = cute.ceil_div(M, block_size)
    grid_y = cute.ceil_div(N, block_size)
    
    gemm_mish_mish_kernel(mA, mB, mC, M, N, K).launch(
        grid=(grid_x, grid_y, 1), 
        block=(block_size, block_size, 1)
    )

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        self.compiled = {}

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Prepare input tensor
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Prepare weight tensor (transposed for matmul)
        weight_t = self.weight.t().contiguous()
        mB = from_dlpack(weight_t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Output tensor
        output = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile key
        key = (x.dtype, batch_size, self.in_features, self.out_features)
        compiled = self.compiled.get(key)
        
        if compiled is None:
            compiled = cute.compile(gemm_mish_mish_host, mA, mB, mC, batch_size, self.out_features, self.in_features)
            self.compiled[key] = compiled
        
        # Run fused kernel
        compiled(mA, mB, mC, batch_size, self.out_features, self.in_features)
        
        # Add bias
        output += self.bias
        
        return output