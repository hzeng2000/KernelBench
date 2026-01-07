import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_post_conv_kernel(gY: cute.Tensor, gBias: cute.Tensor, gZ: cute.Tensor): 
    tidx, _, _ = cute.arch.thread_idx()  
    bidx, _, _ = cute.arch.block_idx()  
    bdim, _, _ = cute.arch.block_dim()  

    thread_idx = bidx * bdim + tidx

    B, C, D, H, W = gY.shape
    total_elems = B * C * D * H * W
    if thread_idx >= total_elems:
        return

    cd_hw = C * D * H * W
    d_hw = D * H * W
    h_w = H * W

    b = thread_idx // cd_hw
    c = (thread_idx % cd_hw) // d_hw
    d = (thread_idx % d_hw) // h_w
    h = (thread_idx % h_w) // W
    w = thread_idx % W

    y_val = gY[b, c, d, h, w]
    bias_val = gBias[c, 0, 0, 0]

    z_val = ((y_val + bias_val + y_val) * y_val) + y_val
    gZ[b, c, d, h, w] = z_val

@cute.jit
def fused_post_conv_host(mY: cute.Tensor, mBias: cute.Tensor, mZ: cute.Tensor):
    B, C, D, H, W = mY.shape
    total_elems = B * C * D * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_post_conv_kernel(mY, mBias, mZ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by fused element-wise operations using a custom CuTe kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous().cuda()
        B, C, D, H, W = x.shape
        z = torch.empty((B, C, D, H, W), dtype=x.dtype, device=x.device)

        mY = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mZ = from_dlpack(z, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_post_conv_host, mY, mBias, mZ)
            self.compiled[key] = compiled

        compiled(mY, mBias, mZ)
        return z