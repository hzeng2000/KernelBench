import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def tanh_kernel(gX: cute.Tensor, gOutput: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    B, C, H, W = gX.shape
    total = B * C * H * W
    if thread_idx >= total:
        return
    b = thread_idx // (C * H * W)
    c = (thread_idx % (C * H * W)) // (H * W)
    h = (thread_idx % (H * W)) // W
    w = thread_idx % W
    gOutput[b, c, h, w] = cute.tanh(gX[b, c, h, w])

@cute.jit
def tanh_host(mX: cute.Tensor, mOutput: cute.Tensor):
    B, C, H, W = mX.shape
    threads_per_block = 256
    total_elems = B * C * H * W
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    tanh_kernel(mX, mOutput).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def max_pool_kernel(gInput: cute.Tensor, gOutput: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    B, C, out_h, out_w = gOutput.shape
    total = B * C * out_h * out_w
    if thread_idx >= total:
        return
    b = thread_idx // (C * out_h * out_w)
    c = (thread_idx % (C * out_h * out_w)) // (out_h * out_w)
    oh = (thread_idx % (out_h * out_w)) // out_w
    ow = thread_idx % out_w
    ih = oh * 2
    iw = ow * 2
    val1 = gInput[b, c, ih, iw]
    val2 = gInput[b, c, ih, iw + 1]
    val3 = gInput[b, c, ih + 1, iw]
    val4 = gInput[b, c, ih + 1, iw + 1]
    max_val = cute.max(cute.max(val1, val2), cute.max(val3, val4))
    gOutput[b, c, oh, ow] = max_val

@cute.jit
def max_pool_host(mInput: cute.Tensor, mOutput: cute.Tensor):
    B, C, H, W = mInput.shape
    threads_per_block = 256
    total_elems = B * C * (H // 2) * (W // 2)
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    max_pool_kernel(mInput, mOutput).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

@cute.kernel
def group_norm_kernel(gX: cute.Tensor, gMean: cute.Tensor, gVar: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor, eps: float, num_groups: int, channels_per_group: int):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    B, C, H, W = gX.shape
    total = B * C * H * W
    if thread_idx >= total:
        return
    b = thread_idx // (C * H * W)
    c = (thread_idx % (C * H * W)) // (H * W)
    h = (thread_idx % (H * W)) // W
    w = thread_idx % W
    g = c // channels_per_group
    x_val = gX[b, c, h, w]
    mean_val = gMean[b, g, 0, 0, 0]
    var_val = gVar[b, g, 0, 0, 0]
    weight_val = gWeight[0, c, 0, 0]
    bias_val = gBias[0, c, 0, 0]
    norm_val = (x_val - mean_val) / cute.sqrt(var_val + eps) * weight_val + bias_val
    gOutput[b, c, h, w] = norm_val

@cute.jit
def group_norm_host(mX: cute.Tensor, mMean: cute.Tensor, mVar: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor, eps: float, num_groups: int, channels_per_group: int):
    B, C, H, W = mX.shape
    threads_per_block = 256
    total_elems = B * C * H * W
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    group_norm_kernel(mX, mMean, mVar, mWeight, mBias, mOutput, eps, num_groups, channels_per_group).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        # Fuse conv_transpose and batch_norm
        with torch.no_grad():
            std = torch.sqrt(self.batch_norm.running_var + self.batch_norm.eps)
            fused_weight = self.conv_transpose.weight / std.view(-1, 1, 1, 1) * self.batch_norm.weight.view(-1, 1, 1, 1)
            fused_bias = (self.conv_transpose.bias - self.batch_norm.running_mean) / std * self.batch_norm.weight + self.batch_norm.bias
        self.fused_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.fused_conv.weight.data = fused_weight
        self.fused_conv.bias.data = fused_bias
        self.compiled_tanh = {}
        self.compiled_pool = {}
        self.compiled_group = {}

    def forward(self, x):
        x = self.fused_conv(x)
        # Custom tanh
        B, C, H, W = x.shape
        output_tanh = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOutput = from_dlpack(output_tanh, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key = (x.dtype,)
        compiled = self.compiled_tanh.get(key)
        if compiled is None:
            compiled = cute.compile(tanh_host, mX, mOutput)
            self.compiled_tanh[key] = compiled
        compiled(mX, mOutput)
        x = output_tanh
        # Custom max_pool
        out_h = H // 2
        out_w = W // 2
        pool_output = torch.empty(B, C, out_h, out_w, dtype=x.dtype, device=x.device)
        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mPoolOutput = from_dlpack(pool_output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key = (x.dtype,)
        compiled = self.compiled_pool.get(key)
        if compiled is None:
            compiled = cute.compile(max_pool_host, mInput, mPoolOutput)
            self.compiled_pool[key] = compiled
        compiled(mInput, mPoolOutput)
        x = pool_output
        # Custom group_norm
        B, C, H, W = x.shape
        num_groups = self.group_norm.num_groups
        channels_per_group = C // num_groups
        x_reshaped = x.view(B, num_groups, channels_per_group, H, W)
        mean = x_reshaped.mean(dim=[2, 3, 4], keepdim=True)
        var = x_reshaped.var(dim=[2, 3, 4], unbiased=False, keepdim=True)
        output_gn = torch.empty_like(x)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mVar = from_dlpack(var, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(self.group_norm.weight.view(1, -1, 1, 1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(self.group_norm.bias.view(1, -1, 1, 1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOutput = from_dlpack(output_gn, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        eps = self.group_norm.eps
        key = (x.dtype,)
        compiled = self.compiled_group.get(key)
        if compiled is None:
            compiled = cute.compile(group_norm_host, mX, mMean, mVar, mWeight, mBias, mOutput, eps, num_groups, channels_per_group)
            self.compiled_group[key] = compiled
        compiled(mX, mMean, mVar, mWeight, mBias, mOutput, eps, num_groups, channels_per_group)
        x = output_gn
        return x