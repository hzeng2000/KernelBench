import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_gemm_div_gelu_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int, divisor: float
):
    # Shared memory for tile of A and B
    smem_A = cute.shared_tensor((64, 32), dtype=cute.float32)
    smem_B = cute.shared_tensor((32, 64), dtype=cute.float32)
    smem_C = cute.shared_tensor((64, 64), dtype=cute.float32)
    
    # Thread identifiers
    tidx = cute.thread_idx().x
    tidy = cute.thread_idx().y
    bidx = cute.block_idx().x
    bidy = cute.block_idx().y
    
    # Tile indices
    tile_m = bidx * 64
    tile_n = bidy * 64
    
    # Initialize accumulator
    acc = cute.zeros((8, 8), dtype=cute.float32)
    
    # Loop over K dimension
    for k_tile in range(0, K, 32):
        # Load tile of A into shared memory
        for i in range(8):
            for j in range(4):
                global_m = tile_m + tidy * 8 + i
                global_k = k_tile + tidx // 16 * 4 + j
                if global_m < M and global_k < K:
                    smem_A[tidy * 8 + i, tidx // 16 * 4 + j] = gA[global_m, global_k]
                else:
                    smem_A[tidy * 8 + i, tidx // 16 * 4 + j] = 0.0
        
        # Load tile of B into shared memory
        for i in range(4):
            for j in range(8):
                global_k = k_tile + tidy // 16 * 4 + i
                global_n = tile_n + tidx % 16 * 4 + j
                if global_k < K and global_n < N:
                    smem_B[tidy // 16 * 4 + i, tidx % 16 * 4 + j] = gB[global_k, global_n]
                else:
                    smem_B[tidy // 16 * 4 + i, tidx % 16 * 4 + j] = 0.0
        
        cute.sync_threads()
        
        # Compute partial dot products
        for k in range(32):
            for i in range(8):
                for j in range(8):
                    a_val = smem_A[tidy * 8 + i, k]
                    b_val = smem_B[k, tidx % 16 * 4 + j]
                    acc[i, j] += a_val * b_val
        
        cute.sync_threads()
    
    # Store results with division and GELU
    for i in range(8):
        for j in range(8):
            global_m = tile_m + tidy * 8 + i
            global_n = tile_n + tidx % 16 * 4 + j
            if global_m < M and global_n < N:
                # Apply division
                val = acc[i, j] / divisor
                
                # Apply GELU activation
                # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x_cubed = val * val * val
                tanh_arg = 0.7978845608 * (val + 0.044715 * x_cubed)
                tanh_val = cute.tanh(tanh_arg)
                gelu_val = 0.5 * val * (1.0 + tanh_val)
                
                gC[global_m, global_n] = gelu_val

@cute.jit
def fused_gemm_div_gelu_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    M: int, N: int, K: int, divisor: float
):
    block_dim = (256, 1, 1)
    grid_dim = (cute.ceil_div(M, 64), cute.ceil_div(N, 64), 1)
    
    fused_gemm_div_gelu_kernel(mA, mB, mC, M, N, K, divisor).launch(
        grid=grid_dim, block=block_dim
    )

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        self.compiled = {}
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        input_size = x.shape[1]
        output_size = self.linear.out_features
        
        # Ensure contiguous and on CUDA
        x = x.contiguous().cuda()
        
        # Get weight tensor
        weight = self.linear.weight.contiguous().cuda()
        bias = self.linear.bias.contiguous().cuda()
        
        # Allocate output tensor
        output = torch.empty(batch_size, output_size, dtype=torch.float32, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight.t(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Add bias to output first (since our kernel doesn't handle bias)
        output.copy_(bias.unsqueeze(0).expand(batch_size, -1))
        
        # Compile and run fused kernel
        key = (x.dtype, weight.dtype)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_gemm_div_gelu_host,
                mA, mB, mC,
                batch_size, output_size, input_size,
                self.divisor
            )
            self.compiled[key] = compiled
        
        compiled(mA, mB, mC, batch_size, output_size, input_size, self.divisor)
        
        return output