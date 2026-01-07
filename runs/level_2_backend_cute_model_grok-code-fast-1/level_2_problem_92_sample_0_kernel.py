import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_activation_add_kernel(gX_norm: cute.Tensor, gX_conv: cute.Tensor, gX_res: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    batch, out_channels, height, width = gX_norm.shape
    total_elems = batch * out_channels * height * width

    if thread_idx >= total_elems:
        return

    b = thread_idx // (out_channels * height * width)
    c = (thread_idx // (height * width)) % out_channels
    h = (thread_idx // width) % height
    w = thread_idx % width

    x_norm_val = gX_norm[b, c, h, w]
    x_conv_val = gX_conv[b, c, h, w]

    tanh_val = cute.tanh(x_norm_val)
    hardswish_val = tanh_val * cute.clamp(tanh_val + 3.0, 0.0, 6.0) / 6.0

    gX_res[b, c, h, w] = x_conv_val + hardswish_val

@cute.jit
def fused_activation_add_host(mX_norm: cute.Tensor, mX_conv: cute.Tensor, mX_res: cute.Tensor):
    batch, out_channels, height, width = mX_norm.shape
    total_elems = batch * out_channels * height * width

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_activation_add_kernel(mX_norm, mX_conv, mX_res).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = torch.nn.GroupNorm(groups, out_channels, eps=eps)
        self.compiled = {}

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Fused Tanh + HardSwish + Residual Addition
        batch, out_channels, height, width = x_norm.shape
        x_norm = x_norm.contiguous().cuda()
        x_conv = x_conv.contiguous().cuda()
        x_res = torch.empty((batch, out_channels, height, width), dtype=x_norm.dtype, device=x_norm.device)

        mX_norm = from_dlpack(x_norm, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mX_conv = from_dlpack(x_conv, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mX_res = from_dlpack(x_res, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x_norm.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_activation_add_host, mX_norm, mX_conv, mX_res)
            self.compiled[key] = compiled

        compiled(mX_norm, mX_conv, mX_res)
        # LogSumExp
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp