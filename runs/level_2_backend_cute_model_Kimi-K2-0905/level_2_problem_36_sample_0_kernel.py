import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_transpose_min_sum_gelu_add_kernel(
    gX: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor,
    N: int, H: int, W: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    global_tid = (bidz * cute.arch.grid_dim().y * cute.arch.grid_dim().x +
                  bidy * cute.arch.grid_dim().x + bidx) * (bdimx * bdimy * bdimz) +
                 tidz * (bdimx * bdimy) + tidy * bdimx + tidx

    total_threads = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z * (bdimx * bdimy * bdimz)
    
    for idx in range(global_tid, N * H * W, total_threads):
        n = idx // (H * W)
        hw = idx % (H * W)
        h = hw // W
        w = hw % W

        min_val = gX[n, 0, h, w]
        for c in range(1, gX.shape[1]):
            val = gX[n, c, h, w]
            if val < min_val:
                min_val = val

        sum_val = 0.0
        for h_idx in range(H):
            sum_val += min_val

        gelu_val = 0.5 * sum_val * (1.0 + cute.math.tanh(0.7978845608 * (sum_val + 0.044715 * sum_val * sum_val * sum_val)))

        gOut[n, 0, 0, w] = gelu_val + gBias[0, 0, 0, w]

@cute.jit
def fused_transpose_min_sum_gelu_add_host(
    mX: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor,
    N: int, H: int, W: int
):
    threads_per_block = 256
    total_elems = N * H * W
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_transpose_min_sum_gelu_add_kernel(
        mX, mBias, mOut, N, H, W
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        N, C, H, W = x.shape
        
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        
        out = torch.empty((N, 1, 1, W), dtype=x.dtype, device=x.device)
        
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_transpose_min_sum_gelu_add_host, mX, mBias, mOut, N, H, W)
            self.compiled[key] = compiled
            
        compiled(mX, mBias, mOut, N, H, W)
        return out