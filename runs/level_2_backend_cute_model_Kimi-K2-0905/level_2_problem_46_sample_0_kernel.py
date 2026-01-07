import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_sub_tanh_sub_pool_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    N: int, H: int, W: int, C_in: int, C_out: int,
    subtract1: float, subtract2: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    # Output spatial dimensions after conv (stride=1, pad=1) and pooling
    H_out = H
    W_out = W
    H_pool = H_out // 2
    W_pool = W_out // 2

    # Each thread computes one output pixel in the pooled output
    out_n = bidx
    out_c = bidy * bdimy + tidy
    out_h = bidz * bdimz + tidz
    out_w = tidx

    if out_n < N and out_c < C_out and out_h < H_pool and out_w < W_pool:
        # Compute convolution for 2x2 pooling window
        sum_val = 0.0
        for kh in range(3):
            for kw in range(3):
                h_in = out_h * 2 - 1 + kh
                w_in = out_w * 2 - 1 + kw
                if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    for c in range(C_in):
                        x_val = gX[out_n, c, h_in, w_in]
                        w_val = gW[out_c, c, kh, kw]
                        sum_val += x_val * w_val
        sum_val += gB[out_c]
        
        # Subtract 1, tanh, subtract 2
        val = sum_val - subtract1
        val = cute.math.tanh(val)
        val = val - subtract2
        
        # Average pooling (already averaged over 2x2 window in conv)
        gY[out_n, out_c, out_h, out_w] = val * 0.25

@cute.jit
def conv_sub_tanh_sub_pool_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    subtract1: float, subtract2: float
):
    N, C_in, H, W = mX.shape
    C_out = mW.shape[0]

    threads_per_block = (8, 8, 8)
    grid_x = cute.ceil_div(N, 1)
    grid_y = cute.ceil_div(C_out, threads_per_block[1])
    grid_z = cute.ceil_div(H // 2, threads_per_block[2])

    conv_sub_tanh_sub_pool_kernel(
        mX, mW, mB, mY, N, H, W, C_in, C_out, subtract1, subtract2
    ).launch(
        grid=(grid_x, grid_y, grid_z),
        block=threads_per_block
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.compiled = {}

    def forward(self, x):
        N, C_in, H, W = x.shape
        C_out = self.conv.out_channels
        
        # Ensure contiguous and on CUDA
        x = x.contiguous().cuda()
        
        # Get weight and bias
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()
        
        # Output tensor
        y = torch.empty(N, C_out, H//2, W//2, dtype=x.dtype, device=x.device)
        
        # Convert to CuTe tensors
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        # Compile and run kernel
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_sub_tanh_sub_pool_host,
                mX, mW, mB, mY,
                self.subtract1_value, self.subtract2_value
            )
            self.compiled[key] = compiled
        
        compiled(mX, mW, mB, mY, self.subtract1_value, self.subtract2_value)
        return y