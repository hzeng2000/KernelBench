import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def post_process_kernel(gX: cute.Tensor, gSubtract: cute.Tensor, gOut: cute.Tensor, B: int, D: int, H: int, W: int):
    C = gSubtract.shape[0]
    bidx = cute.arch.block_idx().x
    tidx = cute.arch.thread_idx().x
    spatial_idx = bidx
    c = tidx
    total_spatial = B * D * H * W
    if spatial_idx >= total_spatial or c >= C:
        return
    temp = spatial_idx
    b = temp // (D * H * W)
    temp = temp % (D * H * W)
    d = temp // (H * W)
    temp = temp % (H * W)
    h = temp // W
    w = temp % W
    x_val = gX[b, c, d, h, w]
    __shared__ float shared_x[16]
    shared_x[tidx] = x_val
    cute.arch.syncthreads()
    if tidx == 0:
        float sum_exp = 0.0f
        for (int i = 0; i < C; i++):
            sum_exp += __expf(shared_x[i])
        for (int i = 0; i < C; i++):
            shared_x[i] = __expf(shared_x[i]) / sum_exp
    cute.arch.syncthreads()
    float val = shared_x[tidx]
    val -= gSubtract[c]
    val = (1.0f / (1.0f + __expf(-val))) * val
    __shared__ float shared_max[16]
    shared_max[tidx] = val
    cute.arch.syncthreads()
    if tidx == 0:
        float max_val = shared_max[0]
        for (int i = 1; i < C; i++):
            max_val = max(max_val, shared_max[i])
        gOut[b, d, h, w] = max_val

@cute.jit
def post_process_host(mX: cute.Tensor, mSubtract: cute.Tensor, mOut: cute.Tensor, B: int, D: int, H: int, W: int):
    C = mSubtract.shape[0]
    total_spatial = B * D * H * W
    threads_per_block = C
    grid_x = total_spatial
    post_process_kernel(mX, mSubtract, mOut, B, D, H, W).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = cutlass.ops.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, conv_mode=cutlass.ConvMode.Deconv)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        B, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        subtract = self.subtract.contiguous().cuda()
        out = torch.empty((B, D, H, W), dtype=x.dtype, device=x.device)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mSubtract = from_dlpack(subtract, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(post_process_host, mX, mSubtract, mOut, B, D, H, W)
            self.compiled[key] = compiled
        compiled(mX, mSubtract, mOut, B, D, H, W)
        return out