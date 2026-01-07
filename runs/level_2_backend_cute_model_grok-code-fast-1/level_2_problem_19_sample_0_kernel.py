import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def gelu_kernel(gA: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    B, C, H, W = gA.shape
    total_elems = B * C * H * W
    if thread_idx < total_elems:
        idx = thread_idx
        n = idx % W
        idx //= W
        h = idx % H
        idx //= H
        c = idx % C
        b = idx // C
        a_val = gA[b, c, h, n]
        gC[b, c, h, n] = 0.5 * a_val * (1 + erf(a_val / 1.41421356237))

@cute.jit
def gelu_host(mA: cute.Tensor, mC: cute.Tensor):
    B, C, H, W = mA.shape
    total_elems = B * C * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    gelu_kernel(mA, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def group_norm_reduce_kernel(gA: cute.Tensor, gSum: cute.Tensor, gSumsq: cute.Tensor, num_groups, cpg, H, W):
    bidx = cute.arch.block_idx(0)
    batch_idx = bidx // num_groups
    group_idx = bidx % num_groups
    tidx = cute.arch.thread_idx(0)
    num_threads = cute.arch.block_dim(0)
    num_elements = cpg * H * W
    elements_per_thread = cute.ceil_div(num_elements, num_threads)
    start = tidx * elements_per_thread
    end = min(start + elements_per_thread, num_elements)
    sum_val = 0.0
    sumsq_val = 0.0
    for i in range(start, end):
        c = i % cpg
        hw = i // cpg
        h = hw % H
        w = hw // H
        val = gA[batch_idx, group_idx * cpg + c, h, w]
        sum_val += val
        sumsq_val += val * val
    cute.atomic.add(gSum[batch_idx, group_idx], sum_val)
    cute.atomic.add(gSumsq[batch_idx, group_idx], sumsq_val)

@cute.jit
def group_norm_reduce_host(mA: cute.Tensor, mSum: cute.Tensor, mSumsq: cute.Tensor, num_groups, cpg, H, W):
    B = mA.shape[0]
    grid_x = B * num_groups
    threads_per_block = 256
    group_norm_reduce_kernel(mA, mSum, mSumsq, num_groups, cpg, H, W).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def group_norm_normalize_kernel(gA: cute.Tensor, gC: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor, gGamma: cute.Tensor, gBeta: cute.Tensor, eps, num_groups, cpg, H, W):
    bidx = cute.arch.block_idx(0)
    batch_idx = bidx // num_groups
    group_idx = bidx % num_groups
    tidx = cute.arch.thread_idx(0)
    num_threads = cute.arch.block_dim(0)
    num_elements = cpg * H * W
    elements_per_thread = cute.ceil_div(num_elements, num_threads)
    start = tidx * elements_per_thread
    end = min(start + elements_per_thread, num_elements)
    mean = gMean[batch_idx, group_idx]
    var = gVar[batch_idx, group_idx]
    for i in range(start, end):
        c = i % cpg
        hw = i // cpg
        h = hw % H
        w = hw // H
        val = gA[batch_idx, group_idx * cpg + c, h, w]
        gamma = gGamma[group_idx * cpg + c]
        beta = gBeta[group_idx * cpg + c]
        gC[batch_idx, group_idx * cpg + c, h, w] = (val - mean) / sqrt(var + eps) * gamma + beta

@cute.jit
def group_norm_normalize_host(mA: cute.Tensor, mC: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor, mGamma: cute.Tensor, mBeta: cute.Tensor, eps, num_groups, cpg, H, W):
    B = mA.shape[0]
    grid_x = B * num_groups
    threads_per_block = 256
    group_norm_normalize_kernel(mA, mC, mMean, mVar, mGamma, mBeta, eps, num_groups, cpg, H, W).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.compiled_gelu = {}
        self.compiled_reduce = {}
        self.compiled_normalize = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, H, W = x.shape
        x = x.contiguous().cuda()
        C_out = C
        num_groups = self.group_norm.num_groups
        cpg = C_out // num_groups
        eps = self.group_norm.eps

        # GELU
        C_gelu = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
        mA_gelu = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC_gelu = from_dlpack(C_gelu, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key_gelu = (x.dtype,)
        compiled_gelu = self.compiled_gelu.get(key_gelu)
        if compiled_gelu is None:
            compiled_gelu = cute.compile(gelu_host, mA_gelu, mC_gelu)
            self.compiled_gelu[key_gelu] = compiled_gelu
        compiled_gelu(mA_gelu, mC_gelu)
        x = C_gelu

        # GroupNorm reduce
        sum_tensor = torch.zeros(B, num_groups, dtype=torch.float32, device=x.device)
        sumsq_tensor = torch.zeros(B, num_groups, dtype=torch.float32, device=x.device)
        mA_reduce = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mSum = from_dlpack(sum_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mSumsq = from_dlpack(sumsq_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        key_reduce = (x.dtype, num_groups, cpg, H, W)
        compiled_reduce = self.compiled_reduce.get(key_reduce)
        if compiled_reduce is None:
            compiled_reduce = cute.compile(group_norm_reduce_host, mA_reduce, mSum, mSumsq, num_groups, cpg, H, W)
            self.compiled_reduce[key_reduce] = compiled_reduce
        compiled_reduce(mA_reduce, mSum, mSumsq)

        # Compute mean and var
        num_elements = cpg * H * W
        mean_tensor = sum_tensor / num_elements
        var_tensor = sumsq_tensor / num_elements - mean_tensor ** 2

        # GroupNorm normalize
        C_norm = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
        mA_norm = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mC_norm = from_dlpack(C_norm, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mMean = from_dlpack(mean_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mVar = from_dlpack(var_tensor, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mGamma = from_dlpack(self.group_norm.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mBeta = from_dlpack(self.group_norm.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        key_normalize = (x.dtype, num_groups, cpg, H, W)
        compiled_normalize = self.compiled_normalize.get(key_normalize)
        if compiled_normalize is None:
            compiled_normalize = cute.compile(group_norm_normalize_host, mA_norm, mC_norm, mMean, mVar, mGamma, mBeta, eps, num_groups, cpg, H, W)
            self.compiled_normalize[key_normalize] = compiled_normalize
        compiled_normalize(mA_norm, mC_norm, mMean, mVar, mGamma, mBeta, eps, num_groups, cpg, H, W)
        return C_norm

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 3
stride = 1
groups = 8
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]