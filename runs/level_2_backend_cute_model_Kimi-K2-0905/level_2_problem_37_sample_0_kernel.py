import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_matmul_swish_bias_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, gBias: cute.Tensor,
    M: int, N: int, K: int
):
    # Shared memory for tile of A and B
    smem_A = cute.shared_tensor((64, 32), dtype=cute.float32)
    smem_B = cute.shared_tensor((32, 64), dtype=cute.float32)
    smem_C = cute.shared_tensor((64, 64), dtype=cute.float32)

    # Thread identifiers
    tid = cute.thread_idx()
    bid = cute.block_idx()

    # Tile indices
    tile_m = bid.x
    tile_n = bid.y

    # Thread-local accumulators
    acc = cute.zeros((8, 8), dtype=cute.float32)

    # Iterate over K dimension
    for k_tile in range(0, K, 32):
        # Load A tile
        for i in range(8):
            for j in range(4):
                global_m = tile_m * 64 + (tid // 4) * 8 + i
                global_k = k_tile + (tid % 4) * 8 + j
                if global_m < M and global_k < K:
                    smem_A[(tid // 4) * 8 + i, (tid % 4) * 8 + j] = gA[global_m, global_k]
        
        # Load B tile
        for i in range(4):
            for j in range(8):
                global_k = k_tile + (tid // 8) * 4 + i
                global_n = tile_n * 64 + (tid % 8) * 8 + j
                if global_k < K and global_n < N:
                    smem_B[(tid // 8) * 4 + i, (tid % 8) * 8 + j] = gB[global_k, global_n]
        
        cute.sync_threads()

        # Compute partial dot products
        for k in range(32):
            for i in range(8):
                for j in range(8):
                    acc[i, j] += smem_A[(tid // 8) * 8 + i, k] * smem_B[k, (tid % 8) * 8 + j]
        
        cute.sync_threads()

    # Store results with Swish and bias
    for i in range(8):
        for j in range(8):
            global_m = tile_m * 64 + (tid // 8) * 8 + i
            global_n = tile_n * 64 + (tid % 8) * 8 + j
            if global_m < M and global_n < N:
                val = acc[i, j] + gBias[global_n]
                sigmoid = 1.0 / (1.0 + cute.exp(-val))
                gC[global_m, global_n] = val * sigmoid

@cute.kernel
def group_norm_kernel(
    gX: cute.Tensor, gY: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    M: int, N: int, num_groups: int
):
    tid = cute.thread_idx()
    bid = cute.block_idx()

    group_size = N // num_groups
    group_id = tid // group_size
    elem_id = tid % group_size

    # Compute group mean
    sum_val = 0.0
    for m in range(M):
        for i in range(group_size):
            sum_val += gX[m, group_id * group_size + i]
    mean = sum_val / (M * group_size)

    # Compute group variance
    var_sum = 0.0
    for m in range(M):
        for i in range(group_size):
            val = gX[m, group_id * group_size + i] - mean
            var_sum += val * val
    var = var_sum / (M * group_size)
    std_dev = cute.sqrt(var + 1e-5)

    # Normalize and apply gamma/beta
    for m in range(M):
        val = gX[m, tid]
        normalized = (val - mean) / std_dev
        gY[m, tid] = normalized * gGamma[tid] + gBeta[tid]

@cute.jit
def fused_matmul_swish_bias_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
    M: int, N: int, K: int
):
    grid = (cute.ceil_div(M, 64), cute.ceil_div(N, 64), 1)
    block = (256, 1, 1)
    fused_matmul_swish_bias_kernel(mA, mB, mC, mBias, M, N, K).launch(grid=grid, block=block)

@cute.jit
def group_norm_host(
    mX: cute.Tensor, mY: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    M: int, N: int, num_groups: int
):
    grid = (1, 1, 1)
    block = (N, 1, 1)
    group_norm_kernel(mX, mY, mGamma, mBeta, M, N, num_groups).launch(grid=grid, block=block)

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        self.out_features = out_features
        
        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
        self.compiled_matmul = None
        self.compiled_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_features = x.shape[1]
        
        # Ensure contiguous and on CUDA
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        gamma = self.gamma.contiguous().cuda()
        beta = self.beta.contiguous().cuda()
        
        # Output tensor
        output = torch.empty((batch_size, self.out_features), dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mOut = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        # Compile and run fused matmul + swish + bias
        if self.compiled_matmul is None:
            self.compiled_matmul = cute.compile(
                fused_matmul_swish_bias_host, 
                mX, mW, mOut, mBias,
                batch_size, self.out_features, in_features
            )
        self.compiled_matmul(mX, mW, mOut, mBias, batch_size, self.out_features, in_features)
        
        # GroupNorm
        norm_output = torch.empty_like(output)
        mNormOut = from_dlpack(norm_output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGamma = from_dlpack(gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        if self.compiled_norm is None:
            self.compiled_norm = cute.compile(
                group_norm_host,
                mOut, mNormOut, mGamma, mBeta,
                batch_size, self.out_features, self.num_groups
            )
        self.compiled_norm(mOut, mNormOut, mGamma, mBeta, batch_size, self.out_features, self.num_groups)
        
        return norm_output