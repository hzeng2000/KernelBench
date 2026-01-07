import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_add_hardswish_kernel(gX: cute.Tensor, gAdd: cute.Tensor, gC: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx
    B, C, D, H, W = gX.shape
    total = B * C * D * H * W
    if thread_idx >= total:
        return
    wi = thread_idx % W
    hi = (thread_idx // W) % H
    di = (thread_idx // (W * H)) % D
    ci = (thread_idx // (W * H * D)) % C
    bi = thread_idx // (W * H * D * C)
    x_val = gX[bi, ci, di, hi, wi]
    add_val = gAdd[bi, ci, di, hi, wi]
    temp = x_val + add_val
    relu6_val = cute.max(cute.min(temp + 3.0, 6.0), 0.0)
    h_val = temp * relu6_val / 6.0
    gC[bi, ci, di, hi, wi] = temp * h_val

@cute.jit
def fused_add_hardswish_host(mX: cute.Tensor, mAdd: cute.Tensor, mC: cute.Tensor):
    B, C, D, H, W = mX.shape
    total_elems = B * C * D * H * W
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    fused_add_hardswish_kernel(mX, mAdd, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        add_input = add_input.contiguous().cuda()
        C_out = torch.empty((B, C, D, H, W), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mAdd = from_dlpack(add_input, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mC = from_dlpack(C_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_add_hardswish_host, mX, mAdd, mC)
            self.compiled[key] = compiled
        compiled(mX, mAdd, mC)
        return C_out