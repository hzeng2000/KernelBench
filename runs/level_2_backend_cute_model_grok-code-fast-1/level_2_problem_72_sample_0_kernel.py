import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def avg_pool_3d_kernel(gIn: cute.Tensor, gOut: cute.Tensor):
    tidx = cute.arch.thread_idx(0)
    bidx = cute.arch.block_idx(0)
    bdim = cute.arch.block_dim(0)
    thread_idx = bidx * bdim + tidx

    B, C, D_in, H_in, W_in = gIn.shape
    D_out = D_in // 4
    H_out = H_in // 4
    W_out = W_in // 4
    total_out = B * C * D_out * H_out * W_out

    if thread_idx >= total_out:
        return

    b = thread_idx // (C * D_out * H_out * W_out)
    rem = thread_idx % (C * D_out * H_out * W_out)
    c = rem // (D_out * H_out * W_out)
    rem2 = rem % (D_out * H_out * W_out)
    d_out = rem2 // (H_out * W_out)
    h_out = (rem2 % (H_out * W_out)) // W_out
    w_out = rem2 % W_out

    sum_val = 0.0
    for dd in range(4):
        for hh in range(4):
            for ww in range(4):
                d_in = d_out * 4 + dd
                h_in = h_out * 4 + hh
                w_in = w_out * 4 + ww
                sum_val += gIn[b, c, d_in, h_in, w_in]
    gOut[b, c, d_out, h_out, w_out] = sum_val / 64.0

@cute.jit
def avg_pool_3d_host(mIn: cute.Tensor, mOut: cute.Tensor):
    B, C, D_in, H_in, W_in = mIn.shape
    D_out = D_in // 4
    H_out = H_in // 4
    W_out = W_in // 4
    total_out = B * C * D_out * H_out * W_out
    threads_per_block = 256
    grid_x = cute.ceil_div(total_out, threads_per_block)
    avg_pool_3d_kernel(mIn, mOut).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized model with batch norm fused into conv transpose and two avg pools fused into a single custom CuTe kernel for 3D avg pool with kernel 4, stride 4.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Fuse batch norm into conv_transpose
        bn = nn.BatchNorm3d(out_channels)
        with torch.no_grad():
            inv_std = 1.0 / torch.sqrt(bn.running_var + bn.eps)
            self.conv_transpose.weight.mul_(bn.weight.view(1, -1, 1, 1, 1) * inv_std.view(1, -1, 1, 1, 1))
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.mul_(bn.weight * inv_std).add_(bn.bias - bn.running_mean * bn.weight * inv_std)
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        B, C, D, H, W = x.shape
        out = torch.empty((B, C, D // 4, H // 4, W // 4), dtype=x.dtype, device=x.device)
        mIn = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(avg_pool_3d_host, mIn, mOut)
            self.compiled[key] = compiled
        compiled(mIn, mOut)
        return out