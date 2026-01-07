import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_matmul_swish_scale_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    M: int, N: int, K: int, scale: float
):
    # Shared memory for tile of A and B
    smem_A = cute.shared_tensor((64, 32), dtype=cute.float32)
    smem_B = cute.shared_tensor((32, 64), dtype=cute.float32)
    smem_C = cute.shared_tensor((64, 64), dtype=cute.float32)

    # Thread-local accumulator
    acc = cute.local_tensor((1, 1), dtype=cute.float32)
    acc[0, 0] = 0.0

    # Thread identifiers
    tidx = cute.thread_idx()
    tidy = cute.thread_idx() // 32
    tidz = cute.thread_idx() % 32

    # Block identifiers
    bx = cute.block_idx()
    by = cute.block_idx() // cute.grid_dim()
    bz = cute.block_idx() % cute.grid_dim()

    # Tile indices
    tile_m = by
    tile_n = bx

    # Global thread indices
    global_m = tile_m * 64 + tidy
    global_n = tile_n * 64 + tidz

    # Compute dot product for this tile
    for k_tile in range(0, K, 32):
        # Load A tile
        if global_m < M and (k_tile + tidz) < K:
            smem_A[tidy, tidz] = gA[global_m, k_tile + tidz]
        else:
            smem_A[tidy, tidz] = 0.0

        # Load B tile
        if (k_tile + tidy) < K and global_n < N:
            smem_B[tidy, tidz] = gB[k_tile + tidy, global_n]
        else:
            smem_B[tidy, tidz] = 0.0

        cute.sync_threads()

        # Compute partial dot product
        for k in range(32):
            acc[0, 0] += smem_A[tidy, k] * smem_B[k, tidz]

        cute.sync_threads()

    # Apply Swish activation and scaling
    if global_m < M and global_n < N:
        val = acc[0, 0]
        sigmoid = 1.0 / (1.0 + cute.exp(-val))
        swish = val * sigmoid
        gC[global_m, global_n] = swish * scale

@cute.jit
def fused_matmul_swish_scale_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    M: int, N: int, K: int, scale: float
):
    grid_m = cute.ceil_div(M, 64)
    grid_n = cute.ceil_div(N, 64)
    fused_matmul_swish_scale_kernel(mA, mB, mC, M, N, K, scale).launch(
        grid=(grid_m * grid_n, 1, 1),
        block=(256, 1, 1)
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
        M, K = x.shape
        N = self.out_features
        
        # Ensure contiguous and on CUDA
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        # Allocate output
        output = torch.empty(M, N, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        
        # Compile kernel if not already done
        key = (x.dtype, self.scaling_factor)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_matmul_swish_scale_host, mA, mB, mC, M, N, K, self.scaling_factor)
            self.compiled[key] = compiled
        
        # Launch kernel
        compiled(mA, mB, mC, M, N, K, self.scaling_factor)
        
        # Add bias
        output += bias.unsqueeze(0)
        
        return output