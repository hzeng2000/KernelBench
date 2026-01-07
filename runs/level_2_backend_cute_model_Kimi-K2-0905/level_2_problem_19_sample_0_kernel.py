import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose_gelu_groupnorm_kernel(
    gI: cute.Tensor, gW: cute.Tensor, gO: cute.Tensor,
    bias: cute.Tensor, running_mean: cute.Tensor, running_var: cute.Tensor,
    N: int, C: int, H: int, W: int, K: int, R: int, S: int, P: int, Q: int,
    stride: int, pad: int, groups: int, num_groups: int, eps: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    global_tid = bidz * bdimx * bdimy * bdimz + tid

    total_threads = cute.arch.grid_dim_x() * cute.arch.grid_dim_y() * cute.arch.grid_dim_z() * bdimx * bdimy * bdimz

    for idx in range(global_tid, N * num_groups * P * Q, total_threads):
        n = idx // (num_groups * P * Q)
        rem = idx % (num_groups * P * Q)
        g = rem // (P * Q)
        rem = rem % (P * Q)
        p = rem // Q
        q = rem % Q

        group_size = C // num_groups
        start_c = g * group_size
        end_c = start_c + group_size

        sum_val = 0.0
        sum_sq = 0.0
        count = 0

        for c in range(start_c, end_c):
            out_val = 0.0
            for kr in range(R):
                for ks in range(S):
                    h_in = p + kr - pad
                    w_in = q + ks - pad
                    if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                        for k in range(K // groups):
                            w_idx = (c // (C // groups)) * (K // groups) + k
                            out_val += gI[n, c, h_in, w_in] * gW[c, w_idx, kr, ks]

            # Apply GELU
            gelu_val = 0.5 * out_val * (1.0 + math.tanh(0.7978845608 * (out_val + 0.044715 * out_val * out_val * out_val)))
            
            # Store intermediate for group norm
            gO[n, c, p, q] = gelu_val
            sum_val += gelu_val
            sum_sq += gelu_val * gelu_val
            count += 1

        # Compute group norm
        mean = sum_val / count
        var = sum_sq / count - mean * mean
        inv_std = 1.0 / math.sqrt(var + eps)

        for c in range(start_c, end_c):
            gO[n, c, p, q] = (gO[n, c, p, q] - mean) * inv_std

@cute.jit
def conv_transpose_gelu_groupnorm_host(
    mI: cute.Tensor, mW: cute.Tensor, mO: cute.Tensor,
    mBias: cute.Tensor, mRunningMean: cute.Tensor, mRunningVar: cute.Tensor,
    N: int, C: int, H: int, W: int, K: int, R: int, S: int, P: int, Q: int,
    stride: int, pad: int, groups: int, num_groups: int, eps: float
):
    threads_per_block = 256
    total_elems = N * num_groups * P * Q
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    conv_transpose_gelu_groupnorm_kernel(
        mI, mW, mO, mBias, mRunningMean, mRunningVar,
        N, C, H, W, K, R, S, P, Q, stride, pad, groups, num_groups, eps
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.compiled = {}
        
    def forward(self, x):
        N, C, H, W = x.shape
        K = self.conv_transpose.out_channels
        R, S = self.conv_transpose.kernel_size
        stride = self.conv_transpose.stride[0]
        pad = self.conv_transpose.padding[0]
        P = (H - 1) * stride - 2 * pad + R
        Q = (W - 1) * stride - 2 * pad + S
        
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        output = torch.empty(N, K, P, Q, dtype=x.dtype, device=x.device)
        
        mI = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mO = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        
        bias = torch.zeros(K, dtype=x.dtype, device=x.device)
        running_mean = torch.zeros(K, dtype=x.dtype, device=x.device)
        running_var = torch.ones(K, dtype=x.dtype, device=x.device)
        
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningMean = from_dlpack(running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningVar = from_dlpack(running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype, C, K, R, stride, self.group_norm.num_groups)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_transpose_gelu_groupnorm_host,
                mI, mW, mO, mBias, mRunningMean, mRunningVar,
                N, C, H, W, K, R, S, P, Q, stride, pad, self.conv_transpose.groups, self.group_norm.num_groups, self.group_norm.eps
            )
            self.compiled[key] = compiled
        
        compiled(mI, mW, mO, mBias, mRunningMean, mRunningVar, N, C, H, W, K, R, S, P, Q, stride, pad, self.conv_transpose.groups, self.group_norm.num_groups, self.group_norm.eps)
        
        # Apply weight and bias from GroupNorm
        output = self.group_norm(output)
        return output