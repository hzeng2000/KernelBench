import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gemm_scale_residual_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    scale: float, M: int, N: int, K: int
):
    # Shared memory for tile of A and B
    smem_A = cute.shared_tensor((64, 64), dtype=cute.float32)
    smem_B = cute.shared_tensor((64, 64), dtype=cute.float32)
    
    # Thread-local accumulators
    accum = cute.local_tensor((1, 1), dtype=cute.float32)
    accum[0, 0] = 0.0
    
    # Thread identifiers
    tid = cute.thread_idx()
    tidx = tid % 16
    tidy = tid // 16
    
    # Block identifiers
    bid_m = cute.block_idx().x
    bid_n = cute.block_idx().y
    
    # Tile indices
    tile_m = bid_m * 64
    tile_n = bid_n * 64
    
    # Iterate over K dimension in tiles
    for tile_k in range(0, K, 64):
        # Load tile of A into shared memory
        if tile_m + tidy < M and tile_k + tidx < K:
            smem_A[tidy, tidx] = gA[tile_m + tidy, tile_k + tidx]
        else:
            smem_A[tidy, tidx] = 0.0
            
        # Load tile of B into shared memory
        if tile_k + tidy < K and tile_n + tidx < N:
            smem_B[tidy, tidx] = gB[tile_k + tidy, tile_n + tidx]
        else:
            smem_B[tidy, tidx] = 0.0
            
        cute.sync_threads()
        
        # Compute partial dot product
        for k in range(64):
            if tile_m + tidy < M and tile_n + tidx < N:
                accum[0, 0] += smem_A[tidy, k] * smem_B[k, tidx]
                
        cute.sync_threads()
    
    # Write result with scaling and residual
    if tile_m + tidy < M and tile_n + tidx < N:
        result = accum[0, 0]
        scaled = result * scale
        gC[tile_m + tidy, tile_n + tidx] = scaled + result

@cute.jit
def gemm_scale_residual_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, scale: float
):
    M, K = mA.shape
    K_b, N = mB.shape
    assert K == K_b
    
    grid_m = cute.ceil_div(M, 64)
    grid_n = cute.ceil_div(N, 64)
    
    gemm_scale_residual_kernel(mA, mB, mC, scale, M, N, K).launch(
        grid=(grid_m, grid_n, 1), block=(256, 1, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        self.in_features = in_features
        self.out_features = out_features
        self.compiled = {}
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / torch.sqrt(torch.tensor(fan_in))
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.weight.t().contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Add bias to output first
        output.copy_(self.bias)
        
        # Compile and launch kernel
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(gemm_scale_residual_host, mA, mB, mC, self.scaling_factor)
            self.compiled[key] = compiled
            
        compiled(mA, mB, mC, self.scaling_factor)
        
        return output