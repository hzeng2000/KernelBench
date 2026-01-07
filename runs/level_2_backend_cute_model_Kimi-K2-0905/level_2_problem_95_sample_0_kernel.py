import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_gemm_bias_swish_tanh_gelu_hardtanh_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, gBias: cute.Tensor,
    alpha: float, beta: float
):
    # Shared memory for tile
    shared_A = cute.shared_tensor((64, 64), dtype=cute.float32)
    shared_B = cute.shared_tensor((64, 64), dtype=cute.float32)
    shared_C = cute.shared_tensor((64, 64), dtype=cute.float32)
    
    # Thread identifiers
    tidx = cute.thread_idx_x()
    tidy = cute.thread_idx_y()
    bidx = cute.block_idx_x()
    bidy = cute.block_idx_y()
    
    # Global thread indices
    row = bidx * 64 + tidy
    col = bidy * 64 + tidx
    
    # Accumulator
    acc = 0.0
    
    # Tile over K dimension
    K = gA.shape[1]
    tiles_k = (K + 63) // 64
    
    for tile_k in range(tiles_k):
        # Load tile of A
        if row < gA.shape[0] and (tile_k * 64 + tidx) < K:
            shared_A[tidy, tidx] = gA[row, tile_k * 64 + tidx]
        else:
            shared_A[tidy, tidx] = 0.0
            
        # Load tile of B
        if (tile_k * 64 + tidy) < K and col < gB.shape[1]:
            shared_B[tidy, tidx] = gB[tile_k * 64 + tidy, col]
        else:
            shared_B[tidy, tidx] = 0.0
            
        cute.sync_threads()
        
        # Compute partial dot product
        for k in range(64):
            acc += shared_A[tidy, k] * shared_B[k, tidx]
            
        cute.sync_threads()
    
    # Add bias and apply activations
    if row < gC.shape[0] and col < gC.shape[1]:
        # Add bias
        acc += gBias[col]
        
        # Swish: x * sigmoid(x)
        sigmoid = 1.0 / (1.0 + cute.exp(-acc))
        swish = acc * sigmoid
        
        # Tanh
        tanh_val = cute.tanh(swish)
        
        # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        gelu_c1 = 0.044715
        gelu_c2 = 0.7978845608  # sqrt(2/pi)
        x_cubed = tanh_val * tanh_val * tanh_val
        gelu_tanh_arg = gelu_c2 * (tanh_val + gelu_c1 * x_cubed)
        gelu = 0.5 * tanh_val * (1.0 + cute.tanh(gelu_tanh_arg))
        
        # Hardtanh: clamp to [-1, 1]
        hardtanh = cute.min(cute.max(gelu, -1.0), 1.0)
        
        gC[row, col] = hardtanh

@cute.jit
def fused_gemm_bias_swish_tanh_gelu_hardtanh_host(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor
):
    M = mA.shape[0]
    N = mB.shape[1]
    
    block_size = 64
    grid_x = (M + block_size - 1) // block_size
    grid_y = (N + block_size - 1) // block_size
    
    fused_gemm_bias_swish_tanh_gelu_hardtanh_kernel(
        mA, mB, mC, mBias, 1.0, 0.0
    ).launch(grid=(grid_x, grid_y, 1), block=(block_size, block_size, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)
        self.compiled = {}
        
    def forward(self, x):
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = self.weight.shape[0]
        
        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        output = torch.empty((batch_size, out_features), dtype=x.dtype, device=x.device)
        
        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(weight.t(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mC = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype, weight.dtype)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_gemm_bias_swish_tanh_gelu_hardtanh_host, mA, mB, mC, mBias)
            self.compiled[key] = compiled
            
        compiled(mA, mB, mC, mBias)
        return output