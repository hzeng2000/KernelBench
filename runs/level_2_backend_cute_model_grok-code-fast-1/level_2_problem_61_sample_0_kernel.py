import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def relu_kernel(gA: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    N, C, Dd, H, W = gA.shape
    total_elems = N * C * Dd * H * W

    if thread_idx < total_elems:
        n = thread_idx // (C * Dd * H * W)
        c = (thread_idx % (C * Dd * H * W)) // (Dd * H * W)
        d = (thread_idx % (Dd * H * W)) // (H * W)
        h = (thread_idx % (H * W)) // W
        w = thread_idx % W

        a_val = gA[n, c, d, h, w]
        gC[n, c, d, h, w] = max(a_val, 0.0)

@cute.jit
def relu_host(mA: cute.Tensor, mC: cute.Tensor):
    N, C, Dd, H, W = mA.shape
    total_elems = N * C * Dd * H * W

    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    relu_kernel(mA, mC).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed 3D convolution, applies custom ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.compiled = {}

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        # Custom ReLU
        N, C, Dd, H, W = x.shape
        x = x.contiguous().cuda()
        out = torch.empty_like(x)

        mA = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mC = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(relu_host, mA, mC)
            self.compiled[key] = compiled

        compiled(mA, mC)
        x = out
        x = self.group_norm(x)
        return x