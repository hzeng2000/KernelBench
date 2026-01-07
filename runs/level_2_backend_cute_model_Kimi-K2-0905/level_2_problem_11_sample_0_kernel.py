import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose_bn_tanh_max_gn_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor,
    gRunningMean: cute.Tensor, gRunningVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    gOutput: cute.Tensor,
    N: int, C_out: int, H_out: int, W_out: int, C_in: int, H_in: int, W_in: int,
    kernel_size: int, stride: int, padding: int, pool_size: int, pool_stride: int, num_groups: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    hw_out = H_out * W_out
    hw_pool = (H_out // 2) * (W_out // 2)

    n = bidx
    c_out = bidy * bdimy + tidy
    hw = bidz * bdimz + tidz

    if n < N and c_out < C_out and hw < hw_out:
        h_out = hw // W_out
        w_out = hw % W_out

        # Compute conv transpose
        acc = 0.0
        for c_in in range(C_in):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    h_in = h_out + padding - kh
                    w_in = w_out + padding - kw
                    if h_in >= 0 and w_in >= 0 and h_in < H_in and w_in < W_in:
                        acc += gInput[n, c_in, h_in, w_in] * gWeight[c_out, c_in, kh, kw]
        acc += gBias[c_out]

        # Batch norm
        mean = gRunningMean[c_out]
        var = gRunningVar[c_out]
        gamma = gGamma[c_out]
        beta = gBeta[c_out]
        bn_out = gamma * (acc - mean) / cute.sqrt(var + 1e-5) + beta

        # Tanh
        tanh_out = cute.tanh(bn_out)

        # Max pool
        h_pool = h_out // pool_stride
        w_pool = w_out // pool_stride
        if h_out % pool_stride == 0 and w_out % pool_stride == 0 and h_pool < H_out // 2 and w_pool < W_out // 2:
            max_val = tanh_out
            for ph in range(pool_size):
                for pw in range(pool_size):
                    if ph == 0 and pw == 0:
                        gOutput[n, c_out, h_pool, w_pool] = max_val

@cute.kernel
def group_norm_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    N: int, C: int, H: int, W: int, num_groups: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    n = bidx
    c = bidy * bdimy + tidy
    hw = bidz * bdimz + tidz
    hw_total = H * W

    if n < N and c < C and hw < hw_total:
        h = hw // W
        w = hw % W

        group_size = C // num_groups
        group_idx = c // group_size
        start_c = group_idx * group_size
        end_c = start_c + group_size

        # Compute mean
        sum_val = 0.0
        for gc in range(start_c, end_c):
            for gh in range(H):
                for gw in range(W):
                    sum_val += gInput[n, gc, gh, gw]
        mean = sum_val / (group_size * H * W)

        # Compute variance
        sum_sq = 0.0
        for gc in range(start_c, end_c):
            for gh in range(H):
                for gw in range(W):
                    diff = gInput[n, gc, gh, gw] - mean
                    sum_sq += diff * diff
        var = sum_sq / (group_size * H * W)

        # Normalize
        gamma = gGamma[c]
        beta = gBeta[c]
        norm_val = gamma * (gInput[n, c, h, w] - mean) / cute.sqrt(var + 1e-5) + beta
        gOutput[n, c, h, w] = norm_val

@cute.jit
def conv_transpose_bn_tanh_max_gn_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor,
    mRunningMean: cute.Tensor, mRunningVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    mOutput: cute.Tensor,
    N: int, C_out: int, H_out: int, W_out: int, C_in: int, H_in: int, W_in: int,
    kernel_size: int, stride: int, padding: int, pool_size: int, pool_stride: int, num_groups: int
):
    threads_per_block = 256
    hw_out = H_out * W_out
    grid_x = N
    grid_y = cute.ceil_div(C_out, 16)
    grid_z = cute.ceil_div(hw_out, 16)

    conv_transpose_bn_tanh_max_gn_kernel(
        mInput, mWeight, mBias, mRunningMean, mRunningVar, mGamma, mBeta, mOutput,
        N, C_out, H_out, W_out, C_in, H_in, W_in,
        kernel_size, stride, padding, pool_size, pool_stride, num_groups
    ).launch(grid=(grid_x, grid_y, grid_z), block=(16, 16, 1))

@cute.jit
def group_norm_host(
    mInput: cute.Tensor, mOutput: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    N: int, C: int, H: int, W: int, num_groups: int
):
    threads_per_block = 256
    hw_total = H * W
    grid_x = N
    grid_y = cute.ceil_div(C, 16)
    grid_z = cute.ceil_div(hw_total, 16)

    group_norm_kernel(
        mInput, mOutput, mGamma, mBeta, N, C, H, W, num_groups
    ).launch(grid=(grid_x, grid_y, grid_z), block=(16, 16, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H_in, W_in = x.shape
        C_out = self.conv_transpose.out_channels
        H_out = (H_in - 1) * self.conv_transpose.stride[0] - 2 * self.conv_transpose.padding[0] + self.conv_transpose.kernel_size[0]
        W_out = (W_in - 1) * self.conv_transpose.stride[1] - 2 * self.conv_transpose.padding[1] + self.conv_transpose.kernel_size[1]
        H_pool, W_pool = H_out // 2, W_out // 2

        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()
        running_mean = self.batch_norm.running_mean.contiguous().cuda()
        running_var = self.batch_norm.running_var.contiguous().cuda()
        bn_weight = self.batch_norm.weight.contiguous().cuda()
        bn_bias = self.batch_norm.bias.contiguous().cuda()

        # Intermediate outputs
        conv_out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
        pool_out = torch.empty((N, C_out, H_pool, W_pool), dtype=x.dtype, device=x.device)
        final_out = torch.empty((N, C_out, H_pool, W_pool), dtype=x.dtype, device=x.device)

        # Convert to CuTe tensors
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningMean = from_dlpack(running_mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mRunningVar = from_dlpack(running_var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGamma = from_dlpack(bn_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(bn_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mConvOut = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mPoolOut = from_dlpack(pool_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mFinalOut = from_dlpack(final_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        # Compile and run fused kernel
        key = (x.dtype, C_in, C_out, H_in, W_in, self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], self.conv_transpose.padding[0])
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_transpose_bn_tanh_max_gn_host, 
                mInput, mWeight, mBias, mRunningMean, mRunningVar, mGamma, mBeta, mConvOut,
                N, C_out, H_out, W_out, C_in, H_in, W_in,
                self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], self.conv_transpose.padding[0], 2, 2, self.group_norm.num_groups)
            self.compiled[key] = compiled

        compiled(mInput, mWeight, mBias, mRunningMean, mRunningVar, mGamma, mBeta, mConvOut,
            N, C_out, H_out, W_out, C_in, H_in, W_in,
            self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], self.conv_transpose.padding[0], 2, 2, self.group_norm.num_groups)

        # Group norm
        gn_weight = self.group_norm.weight.contiguous().cuda()
        gn_bias = self.group_norm.bias.contiguous().cuda()
        mGnGamma = from_dlpack(gn_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mGnBeta = from_dlpack(gn_bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        key_gn = (x.dtype, C_out, H_pool, W_pool, self.group_norm.num_groups)
        compiled_gn = self.compiled.get(key_gn)
        if compiled_gn is None:
            compiled_gn = cute.compile(group_norm_host, mPoolOut, mFinalOut, mGnGamma, mGnBeta,
                N, C_out, H_pool, W_pool, self.group_norm.num_groups)
            self.compiled[key_gn] = compiled_gn

        compiled_gn(mPoolOut, mFinalOut, mGnGamma, mGnBeta,
            N, C_out, H_pool, W_pool, self.group_norm.num_groups)

        return final_out