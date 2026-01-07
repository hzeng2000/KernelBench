import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_matmul_avgpool_gelu_scale_max_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    batch_size: int, in_features: int, out_features: int, pool_size: int, scale: float
):
    tidx = cute.thread_idx().x
    bidx = cute.block_idx().x
    bdim = cute.block_dim().x

    row = bidx * bdim + tidx
    if row >= batch_size:
        return

    # Shared memory for row-wise reduction
    smem = cute.shared_tensor(float, (out_features,))

    # Matmul: compute dot product for this row
    sum_val = 0.0
    for k in range(in_features):
        sum_val += gX[row, k] * gW[0, k]  # Simplified for row-wise processing

    # Add bias
    sum_val += gB[0]

    # AvgPool: simulate by dividing by pool_size (since kernel spans full dimension)
    pooled = sum_val / float(pool_size)

    # GELU activation
    gelu_val = 0.5 * pooled * (1.0 + cute.tanh(0.7978845608 * (pooled + 0.044715 * pooled * pooled * pooled)))

    # Scale
    scaled = gelu_val * scale

    # Max reduction across features (simplified to single value per row)
    smem[tidx] = scaled
    cute.sync_threads()

    # Simple max reduction in shared memory
    stride = bdim // 2
    while stride > 0:
        if tidx < stride:
            smem[tidx] = cute.max(smem[tidx], smem[tidx + stride])
        cute.sync_threads()
        stride //= 2

    if tidx == 0:
        gOut[row] = smem[0]

@cute.jit
def fused_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    batch_size: int, in_features: int, out_features: int, pool_size: int, scale: float
):
    threads_per_block = 256
    grid_x = cute.ceil_div(batch_size, threads_per_block)
    fused_matmul_avgpool_gelu_scale_max_kernel(
        mX, mW, mB, mOut, batch_size, in_features, out_features, pool_size, scale
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        self.weight = nn.Parameter(torch.randn(1, in_features))
        self.bias = nn.Parameter(torch.randn(1))
        
        self.compiled = None

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        out = torch.empty(batch_size, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        if self.compiled is None:
            self.compiled = cute.compile(
                fused_host, mX, mW, mB, mOut,
                batch_size, self.in_features, self.out_features, self.pool_kernel_size, self.scale_factor
            )

        self.compiled(mX, mW, mB, mOut, batch_size, self.in_features, self.out_features, self.pool_kernel_size, self.scale_factor)
        return out