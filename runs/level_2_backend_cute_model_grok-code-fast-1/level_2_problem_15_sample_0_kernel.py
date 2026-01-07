import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def subtract_mean_kernel(gX: cute.Tensor, gMean: cute.Tensor, gOut: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    batch, channels, depth, height, width = gX.shape
    total_elems = batch * channels * depth * height * width

    if thread_idx >= total_elems:
        return

    mi = thread_idx // (channels * depth * height * width)
    ci = (thread_idx % (channels * depth * height * width)) // (depth * height * width)
    di = (thread_idx % (depth * height * width)) // (height * width)
    hi = (thread_idx % (height * width)) // width
    wi = thread_idx % width

    x_val = gX[mi, ci, di, hi, wi]
    mean_val = gMean[mi, ci, 0, 0, 0]
    gOut[mi, ci, di, hi, wi] = x_val - mean_val

@cute.jit
def subtract_mean_host(mX: cute.Tensor, mMean: cute.Tensor, mOut: cute.Tensor):
    batch, channels, depth, height, width = mX.shape
    total_elems = batch * channels * depth * height * width
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    subtract_mean_kernel(mX, mMean, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.compiled = {}

    def forward(self, x):
        weight = self.conv_transpose.weight
        weight_perm = weight.permute(1, 0, 2, 3, 4)
        bias = self.conv_transpose.bias

        # Compute output shape
        if isinstance(kernel_size, int):
            k_d, k_h, k_w = kernel_size, kernel_size, kernel_size
        else:
            k_d, k_h, k_w = kernel_size
        if isinstance(stride, int):
            s_d, s_h, s_w = stride, stride, stride
        else:
            s_d, s_h, s_w = stride
        if isinstance(padding, int):
            p_d, p_h, p_w = padding, padding, padding
        else:
            p_d, p_h, p_w = padding

        batch, in_c, in_d, in_h, in_w = x.shape
        out_d = (in_d - 1) * s_d - 2 * p_d + k_d
        out_h = (in_h - 1) * s_h - 2 * p_h + k_h
        out_w = (in_w - 1) * s_w - 2 * p_w + k_w
        out_shape = (batch, out_channels, out_d, out_h, out_w)

        output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        output = cute.ops.conv3d(x, weight_perm, output, mode="backward_data", stride=stride, padding=padding, dilation=1)
        if bias is not None:
            output += bias.view(1, -1, 1, 1, 1)

        x = self.batch_norm(output)
        mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        out = torch.empty_like(x)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mMean = from_dlpack(mean, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(subtract_mean_host, mX, mMean, mOut)
            self.compiled[key] = compiled

        compiled(mX, mMean, mOut)
        return out