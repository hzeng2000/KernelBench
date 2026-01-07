import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_min_sub_kernel(
    gA: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int, constant: float
):
    # Shared memory for A and B tiles
    smem_A = cute.shared_tensor((64, 32), dtype=cute.float32)
    smem_W = cute.shared_tensor((32, 64), dtype=cute.float32)
    
    # Thread block tile
    tid = cute.thread_idx()
    bid_m = cute.block_idx().x
    bid_n = cute.block_idx().y
    
    # Thread identifiers
    tidx = tid % 8
    tidy = tid // 8
    
    # Global memory access
    row = bid_m * 64 + tidy
    col = bid_n * 64 + tidx
    
    # Accumulator
    acc = 0.0
    
    # Loop over K dimension
    for k_tile in range(0, K, 32):
        # Load A tile
        if row < M and (k_tile + tidx) < K:
            smem_A[tidy, tidx] = gA[row, k_tile + tidx]
        else:
            smem_A[tidy, tidx] = 0.0
        
        # Load W tile
        if (k_tile + tidy) < K and col < N:
            smem_W[tidy, tidx] = gW[k_tile + tidy, col]
        else:
            smem_W[tidy, tidx] = 0.0
        
        cute.sync_threads()
        
        # Compute
        for k in range(32):
            acc += smem_A[tidy, k] * smem_W[k, tidx]
        
        cute.sync_threads()
    
    # Add bias and apply min/sub
    if row < M and col < N:
        acc += gB[col]
        acc = cute.fminf(acc, constant)
        acc = acc - constant
        gC[row, col] = acc

@cute.jit
def fused_gemm_min_sub_host(
    mA: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    M: int, N: int, K: int, constant: float
):
    grid = (cute.ceil_div(M, 64), cute.ceil_div(N, 64), 1)
    block = (64, 1, 1)
    fused_gemm_min_sub_kernel(mA, mW, mB, mC, M, N, K, constant).launch(grid=grid, block=block)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.compiled = {}

    def forward(self, x):
        M = x.shape[0]
        K = self.in_features
        N = self.out_features
        
        # Ensure contiguous and on GPU
        x = x.contiguous().cuda()
        weight = self.linear.weight.data.contiguous().cuda()
        bias = self.linear.bias.data.contiguous().cuda()
        output = torch.empty((M, N), dtype=torch.float32, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile and run kernel
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_min_sub_host, mA, mW, mB, mC, M, N, K, float(self.constant.item()))
            self.compiled[key] = compiled
        
        compiled(mA, mW, mB, mC, M, N, K, float(self.constant.item()))
        return output