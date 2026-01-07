import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def gemm_lse_lrelu_lrelu_gelu_gelu_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int,
    stride_A_m: int, stride_A_k: int,
    stride_B_k: int, stride_B_n: int,
    stride_C_m: int, stride_C_n: int
):
    # Shared memory for tile
    shared_mem = cute.shared_memory(128 * 128 * 4 + 128 * 128 * 4, dtype=cute.float32)
    
    # Thread indices
    tidx = cute.thread_idx().x
    tidy = cute.thread_idx().y
    bidx = cute.block_idx().x
    bidy = cute.block_idx().y
    
    # Tile size
    TILE_M = 128
    TILE_N = 128
    TILE_K = 8
    
    # Global thread indices
    row = bidy * TILE_M + tidy
    col = bidx * TILE_N + tidx
    
    # Accumulator
    acc = cute.constant(0.0, dtype=cute.float32)
    
    # Main GEMM loop
    for k in range(0, K, TILE_K):
        # Load A tile
        if row < M and k + tidx < K:
            a_val = gA[row * stride_A_m + (k + tidx) * stride_A_k]
        else:
            a_val = cute.constant(0.0, dtype=cute.float32)
        
        # Load B tile
        if k + tidy < K and col < N:
            b_val = gB[(k + tidy) * stride_B_k + col * stride_B_n]
        else:
            b_val = cute.constant(0.0, dtype=cute.float32)
        
        # Compute partial dot product
        acc += a_val * b_val
    
    # Apply bias if present
    if row < M and col < N:
        # LogSumExp
        max_val = acc
        exp_sum = cute.exp(acc - max_val)
        lse = max_val + cute.log(exp_sum)
        
        # First LeakyReLU
        lrelu1 = cute.select(lse > 0, lse, lse * 0.01)
        
        # Second LeakyReLU
        lrelu2 = cute.select(lrelu1 > 0, lrelu1, lrelu1 * 0.01)
        
        # First GELU
        gelu1 = 0.5 * lrelu2 * (1.0 + cute.tanh(0.7978845608 * (lrelu2 + 0.044715 * lrelu2 * lrelu2 * lrelu2)))
        
        # Second GELU
        gelu2 = 0.5 * gelu1 * (1.0 + cute.tanh(0.7978845608 * (gelu1 + 0.044715 * gelu1 * gelu1 * gelu1)))
        
        gC[row * stride_C_m + col * stride_C_n] = gelu2

@cute.jit
def gemm_lse_lrelu_lrelu_gelu_gelu_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    M: int, N: int, K: int
):
    threads_per_block = (16, 16)
    blocks_per_grid = (
        cute.ceil_div(N, 128),
        cute.ceil_div(M, 128)
    )
    
    gemm_lse_lrelu_lrelu_gelu_gelu_kernel(
        mA, mB, mC, M, N, K,
        mA.stride(0), mA.stride(1),
        mB.stride(0), mB.stride(1),
        mC.stride(0), mC.stride(1)
    ).launch(grid=blocks_per_grid, block=threads_per_block)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.compiled = {}
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        M = batch_size
        N = self.out_features
        K = self.in_features
        
        # Prepare tensors
        x_contig = x.contiguous().cuda()
        weight_contig = self.weight.contiguous().cuda()
        output = torch.empty(batch_size, self.out_features, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight_contig, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile and launch kernel
        key = (x.dtype, self.bias is not None)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(gemm_lse_lrelu_lrelu_gelu_gelu_host, mA, mB, mC, M, N, K)
            self.compiled[key] = compiled
        
        compiled(mA, mB, mC, M, N, K)
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.unsqueeze(0)
        
        return output