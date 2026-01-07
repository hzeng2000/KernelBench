import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_kernel(gX: cute.Tensor, gBias: cute.Tensor, gOut: cute.Tensor, scaling_factor: float):
    B, C, D, H, W = gX.shape
    _, _, _, _, _ = gBias.shape
    _, _, _, _, _ = gOut.shape

    bhw = cute.arch.block_idx().x
    c = cute.arch.thread_idx().x

    b = bhw // (H * W)
    hw = bhw % (H * W)
    h = hw // W
    w = hw % W

    # Compute mean over D
    mean_val = 0.0
    for d in range(D):
        mean_val += gX[b, c, d, h, w]
    mean_val /= D

    val = mean_val + gBias[0, c, 0, 0, 0]

    # Shared memory for softmax
    shared_val = cute.shared_memory(float, (64,))
    shared_val[c] = val
    cute.arch.syncthreads()

    if c == 0:
        max_val = shared_val[0]
        for i in range(1, C):
            max_val = max(max_val, shared_val[i])
        sum_exp = 0.0
        for i in range(C):
            sum_exp += cute.exp(shared_val[i] - max_val)
        shared_max = cute.shared_memory(float, (1,))
        shared_sum = cute.shared_memory(float, (1,))
        shared_max[0] = max_val
        shared_sum[0] = sum_exp
    cute.arch.syncthreads()

    max_val = shared_max[0]
    sum_exp = shared_sum[0]

    out_val = cute.exp(shared_val[c] - max_val) / sum_exp
    out_val = cute.tanh(out_val)
    out_val *= scaling_factor

    gOut[b, c, 0, h, w] = out_val

@cute.jit
def fused_host(mX: cute.Tensor, mBias: cute.Tensor, mOut: cute.Tensor, scaling_factor: float):
    B, C, D, H, W = mX.shape

    threads_per_block = 64  # One per channel
    blocks = B * H * W

    fused_kernel(mX, mBias, mOut, scaling_factor).launch(grid=(blocks, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.compiled = {}

    def forward(self, x):
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        B, C, D, H, W = x.shape
        x = x.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        out = torch.empty((B, C, 1, H, W), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_host, mX, mBias, mOut, self.scaling_factor)
            self.compiled[key] = compiled

        compiled(mX, mBias, mOut, self.scaling_factor)
        return out