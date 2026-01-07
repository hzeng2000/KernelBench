import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def conv_transpose_relu_gn_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    gMean: cute.Tensor, gVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    D_in: int, H_in: int, W_in: int,
    D_out: int, H_out: int, W_out: int,
    kernel_size: int, stride: int, padding: int, groups: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    bid = bidz * cute.arch.grid_dim().x * cute.arch.grid_dim().y + bidy * cute.arch.grid_dim().x + bidx
    threads_per_block = bdimx * bdimy * bdimz

    elems_per_thread = cute.ceil_div(batch_size * out_channels * D_out * H_out * W_out, threads_per_block)
    start_idx = bid * threads_per_block * elems_per_thread + tid

    for elem in range(elems_per_thread):
        idx = start_idx + elem * threads_per_block
        if idx >= batch_size * out_channels * D_out * H_out * W_out:
            continue

        n = idx // (out_channels * D_out * H_out * W_out)
        rem = idx % (out_channels * D_out * H_out * W_out)
        c_out = rem // (D_out * H_out * W_out)
        rem = rem % (D_out * H_out * W_out)
        d_out = rem // (H_out * W_out)
        rem = rem % (H_out * W_out)
        h_out = rem // W_out
        w_out = rem % W_out

        acc = 0.0
        for c_in in range(in_channels):
            for kd in range(kernel_size):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        d_in = d_out + kd - padding
                        h_in = h_out + kh - padding
                        w_in = w_out + kw - padding
                        if d_in >= 0 and d_in < D_in and h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                            acc += gInput[n, c_in, d_in, h_in, w_in] * gWeight[c_out, c_in, kd, kh, kw]

        if bias:
            acc += gBias[c_out]

        acc = max(acc, 0.0)

        group = c_out // (out_channels // groups)
        group_size = out_channels // groups
        group_start = group * group_size

        sum_val = 0.0
        sum_sq = 0.0
        count = 0
        for gc in range(group_size):
            gc_out = group_start + gc
            for gd in range(D_out):
                for gh in range(H_out):
                    for gw in range(W_out):
                        val = gOutput[n, gc_out, gd, gh, gw] if (n < batch_size and gc_out < out_channels) else 0.0
                        sum_val += val
                        sum_sq += val * val
                        count += 1

        mean = sum_val / count
        var = sum_sq / count - mean * mean
        std = sqrt(var + 1e-5)

        normalized = (acc - mean) / std
        scaled = normalized * gGamma[c_out] + gBeta[c_out]

        gOutput[n, c_out, d_out, h_out, w_out] = scaled

@cute.jit
def conv_transpose_relu_gn_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    mMean: cute.Tensor, mVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    D_in: int, H_in: int, W_in: int,
    D_out: int, H_out: int, W_out: int,
    kernel_size: int, stride: int, padding: int, groups: int, bias: bool
):
    total_elems = batch_size * out_channels * D_out * H_out * W_out
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    conv_transpose_relu_gn_kernel(
        mInput, mWeight, mBias, mOutput,
        mMean, mVar, mGamma, mBeta,
        batch_size, in_channels, out_channels,
        D_in, H_in, W_in, D_out, H_out, W_out,
        kernel_size, stride, padding, groups
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        self.compiled = {}

    def forward(self, x):
        batch_size = x.shape[0]
        D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
        D_out = D_in + self.kernel_size - 1
        H_out = H_in + self.kernel_size - 1
        W_out = W_in + self.kernel_size - 1
        
        x = x.contiguous().cuda()
        output = torch.empty(batch_size, self.out_channels, D_out, H_out, W_out, dtype=x.dtype, device=x.device)
        
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(self.bias if self.bias is not None else torch.zeros(self.out_channels, device=x.device), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        
        mean = torch.zeros(self.groups, device=x.device)
        var = torch.zeros(self.groups, device=x.device)
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        mGamma = from_dlpack(self.gamma, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(self.beta, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        
        key = (x.dtype, batch_size, self.in_channels, self.out_channels, D_in, H_in, W_in)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_transpose_relu_gn_host,
                mInput, mWeight, mBias, mOutput,
                mMean, mVar, mGamma, mBeta,
                batch_size, self.in_channels, self.out_channels,
                D_in, H_in, W_in, D_out, H_out, W_out,
                self.kernel_size, 1, 0, self.groups, self.bias is not None
            )
            self.compiled[key] = compiled
            
        compiled(
            mInput, mWeight, mBias, mOutput,
            mMean, mVar, mGamma, mBeta,
            batch_size, self.in_channels, self.out_channels,
            D_in, H_in, W_in, D_out, H_out, W_out,
            self.kernel_size, 1, 0, self.groups, self.bias is not None
        )
        
        return output