import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_bn_scale_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor, gScale: cute.Tensor, gBias: cute.Tensor,
    gOut: cute.Tensor, scaling_factor: float,
    N: int, C: int, H: int, W: int, K: int, R: int, S: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_id = tidz * bdimx * bdimy + tidy * bdimx + tidx
    block_id = bidz * cute.arch.grid_dim_x() * cute.arch.grid_dim_y() + bidy * cute.arch.grid_dim_x() + bidx

    # Output spatial dimensions
    P = H - R + 1
    Q = W - S + 1

    # Each thread computes one output element
    total_outputs = N * K * P * Q
    global_thread_id = block_id * (bdimx * bdimy * bdimz) + thread_id

    if global_thread_id >= total_outputs:
        return

    # Decompose global thread id into n, k, p, q
    tmp = global_thread_id
    q = tmp % Q
    tmp = tmp // Q
    p = tmp % P
    tmp = tmp // P
    k = tmp % K
    n = tmp // K

    # Compute convolution for this output element
    sum_val = 0.0
    for c in range(C):
        for r in range(R):
            for s in range(S):
                h_in = p + r
                w_in = q + s
                x_val = gX[n, c, h_in, w_in]
                w_val = gW[k, c, r, s]
                sum_val += x_val * w_val

    # Add bias
    sum_val += gB[k]

    # Apply batch normalization
    mean = gMean[k]
    var = gVar[k]
    scale = gScale[k]
    bias = gBias[k]
    
    # BN: (x - mean) / sqrt(var + eps) * scale + bias
    eps = 1e-5
    bn_val = (sum_val - mean) / cute.sqrt(var + eps)
    bn_val = bn_val * scale + bias

    # Apply scaling factor
    out_val = bn_val * scaling_factor

    # Store output
    gOut[n, k, p, q] = out_val

@cute.jit
def conv_bn_scale_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor, mScale: cute.Tensor, mBias: cute.Tensor,
    mOut: cute.Tensor, scaling_factor: float
):
    N, C, H, W = mX.shape
    K, _, R, S = mW.shape
    
    P = H - R + 1
    Q = W - S + 1

    total_outputs = N * K * P * Q
    threads_per_block = 256
    blocks = (total_outputs + threads_per_block - 1) // threads_per_block

    conv_bn_scale_kernel(
        mX, mW, mB, mMean, mVar, mScale, mBias, mOut, scaling_factor,
        N, C, H, W, K, R, S
    ).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get conv weights and bias
        w = self.conv.weight
        b = self.conv.bias
        
        # Get batch norm parameters
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias

        # Input dimensions
        N, C, H, W = x.shape
        K, _, R, S = w.shape
        P = H - R + 1
        Q = W - S + 1

        # Allocate output tensor
        out = torch.empty((N, K, P, Q), dtype=x.dtype, device=x.device)

        # Convert to CuTe tensors
        mX = from_dlpack(x.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(w.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(b.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mMean = from_dlpack(bn_mean.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mVar = from_dlpack(bn_var.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mScale = from_dlpack(bn_weight.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBias = from_dlpack(bn_bias.contiguous(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        # Compile and launch kernel
        key = (x.dtype, w.dtype)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_bn_scale_host, mX, mW, mB, mMean, mVar, mScale, mBias, mOut, self.scaling_factor)
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mMean, mVar, mScale, mBias, mOut, self.scaling_factor)
        return out