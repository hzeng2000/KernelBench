import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_transpose_scale_maxpool_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    scale: float, maxpool_k: int, out_d: int, out_h: int, out_w: int
):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, tidy, tidz = cute.arch.thread_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    # Compute output position
    ow = bidz * bdimz + tidz
    oh = bidy * bdimy + tidy
    od = bidx * bdimx + tidx

    if od < out_d and oh < out_h and ow < out_w:
        # Max pooling window
        max_val = -float('inf')
        for kd in range(maxpool_k):
            for kh in range(maxpool_k):
                for kw in range(maxpool_k):
                    pd = od * maxpool_k + kd
                    ph = oh * maxpool_k + kh
                    pw = ow * maxpool_k + kw
                    if pd < gOutput.shape[1] and ph < gOutput.shape[2] and pw < gOutput.shape[3]:
                        val = gOutput[0, pd, ph, pw]
                        if val > max_val:
                            max_val = val
        gOutput[0, od, oh, ow] = max_val

@cute.kernel
def global_avg_pool_clamp_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    clamp_min: float, clamp_max: float
):
    bidx = cute.arch.block_idx().x
    tidx = cute.arch.thread_idx().x

    d, h, w = gInput.shape[1], gInput.shape[2], gInput.shape[3]
    total = d * h * w
    sum_val = 0.0

    # Each thread handles multiple elements
    for i in range(tidx, total, cute.arch.block_dim().x):
        z = i // (h * w)
        y = (i % (h * w)) // w
        x = i % w
        sum_val += gInput[bidx, z, y, x]

    # Block-level reduction
    shared = cute.shared_memory(float, 256)
    shared[tidx] = sum_val
    cute.sync_threads()

    # Simple reduction
    s = 1
    while s < cute.arch.block_dim().x:
        if tidx % (2 * s) == 0 and tidx + s < cute.arch.block_dim().x:
            shared[tidx] += shared[tidx + s]
        s *= 2
        cute.sync_threads()

    if tidx == 0:
        avg = shared[0] / total
        avg = max(min(avg, clamp_max), clamp_min)
        gOutput[bidx, 0, 0, 0] = avg

@cute.jit
def conv_transpose_scale_maxpool_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    scale: float, maxpool_k: int
):
    batch, out_c, out_d, out_h, out_w = mOutput.shape
    threads = 8
    conv_transpose_scale_maxpool_kernel(
        mInput, mWeight, mBias, mOutput,
        scale, maxpool_k, out_d, out_h, out_w
    ).launch(
        grid=(cute.ceil_div(out_d, threads), cute.ceil_div(out_h, threads), cute.ceil_div(out_w, threads)),
        block=(threads, threads, threads)
    )

@cute.jit
def global_avg_pool_clamp_host(
    mInput: cute.Tensor, mOutput: cute.Tensor,
    clamp_min: float, clamp_max: float
):
    batch = mInput.shape[0]
    threads = 256
    global_avg_pool_clamp_kernel(
        mInput, mOutput, clamp_min, clamp_max
    ).launch(
        grid=(batch, 1, 1),
        block=(threads, 1, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_k = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.compiled = {}

    def forward(self, x):
        # ConvTranspose3d
        x = self.conv_transpose(x)
        # Scale
        x = x * self.scale
        # MaxPool3d (simplified as kernel-level max)
        batch, c, d, h, w = x.shape
        out_d = d // self.maxpool_k
        out_h = h // self.maxpool_k
        out_w = w // self.maxpool_k
        x_pool = torch.zeros(batch, c, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))
        mW = from_dlpack(self.conv_transpose.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))
        mB = from_dlpack(self.conv_transpose.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mPool = from_dlpack(x_pool, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_transpose_scale_maxpool_host, mX, mW, mB, mPool, self.scale, self.maxpool_k)
            self.compiled[key] = compiled
        compiled(mX, mW, mB, mPool, self.scale, self.maxpool_k)

        # GlobalAvgPool3d + Clamp
        out = torch.zeros(batch, c, 1, 1, 1, dtype=x_pool.dtype, device=x_pool.device)
        mPool_in = from_dlpack(x_pool, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1,2,3,4))

        key2 = (x_pool.dtype,)
        compiled2 = self.compiled.get(key2)
        if compiled2 is None:
            compiled2 = cute.compile(global_avg_pool_clamp_host, mPool_in, mOut, self.clamp_min, self.clamp_max)
            self.compiled[key2] = compiled2
        compiled2(mPool_in, mOut, self.clamp_min, self.clamp_max)

        return out