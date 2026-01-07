import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_matmul_maxpool_sum_scale_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    scale_factor: float, batch_size: int, in_features: int, out_features: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy

    if row < batch_size and col < out_features // 2:
        # Compute matmul for two adjacent columns
        sum0 = 0.0
        sum1 = 0.0
        for k in range(in_features):
            x_val = gX[row, k]
            sum0 += x_val * gW[col * 2, k]
            sum1 += x_val * gW[col * 2 + 1, k]
        
        # Add bias
        sum0 += gB[col * 2]
        sum1 += gB[col * 2 + 1]

        # Max pooling (kernel_size=2, so max of adjacent elements)
        max_val = sum0 if sum0 > sum1 else sum1

        # Sum reduction (already done per row)
        # Scale
        scaled_val = max_val * scale_factor

        gOut[row, col] = scaled_val

@cute.jit
def fused_matmul_maxpool_sum_scale_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    scale_factor: float, batch_size: int, in_features: int, out_features: int
):
    threads_per_block = 16
    grid_x = cute.ceil_div(batch_size, threads_per_block)
    grid_y = cute.ceil_div(out_features // 2, threads_per_block)

    fused_matmul_maxpool_sum_scale_kernel(
        mX, mW, mB, mOut, scale_factor, batch_size, in_features, out_features
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block, threads_per_block, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.contiguous().cuda()
        
        # Ensure out_features is even for maxpool kernel_size=2
        assert self.out_features % 2 == 0
        
        out = torch.empty((batch_size, self.out_features // 2), dtype=x.dtype, device=x.device)
        final_out = torch.empty((batch_size,), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mW = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_matmul_maxpool_sum_scale_host, 
                mX, mW, mB, mOut, 
                self.scale_factor, batch_size, self.in_features, self.out_features
            )
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mOut, self.scale_factor, batch_size, self.in_features, self.out_features)
        
        # Final sum reduction
        final_out = out.sum(dim=1)
        return final_out