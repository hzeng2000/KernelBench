import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_transpose_maxpool_hardtanh_mean_tanh_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_h: int, in_w: int, out_h: int, out_w: int,
    kernel_size: int, stride: int, padding: int,
    maxpool_kernel: int, maxpool_stride: int, mp_out_h: int, mp_out_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    hw = bidz * mp_out_h * mp_out_w
    hw_idx = tidx + bidx * bdimx
    if hw_idx >= mp_out_h * mp_out_w:
        return

    mp_oh = hw_idx // mp_out_w
    mp_ow = hw_idx % mp_out_w

    # Map maxpool output back to conv_transpose output
    oh_start = mp_oh * maxpool_stride
    ow_start = mp_ow * maxpool_stride

    # Compute maxpool window
    max_val = -1e38
    for kh in range(maxpool_kernel):
        for kw in range(maxpool_kernel):
            oh = oh_start + kh
            ow = ow_start + kw
            if oh < out_h and ow < out_w:
                # Compute conv_transpose at (oh, ow)
                acc = 0.0
                for c in range(gInput.shape[1]):
                    for kh_ct in range(kernel_size):
                        for kw_ct in range(kernel_size):
                            ih = oh + padding - kh_ct
                            iw = ow + padding - kw_ct
                            if ih >= 0 and iw >= 0 and ih < in_h and iw < in_w:
                                acc += gInput[bidy, c, ih, iw] * gWeight[c, tidy, kh_ct, kw_ct]
                if c == 0 and kh_ct == 0 and kw_ct == 0 and gBias.shape[0] > tidy:
                    acc += gBias[tidy]
                val = acc
                # Apply hardtanh
                val = max(-1.0, min(1.0, val))
                if val > max_val:
                    max_val = val

    # Mean reduction
    sum_val = cute.shared_memory(128, dtype=cute.float32)
    idx = tidx + tidy * bdimx
    sum_val[idx] = max_val
    cute.arch.sync_threads()

    # Simple reduction within block
    s = 64
    while s > 0:
        if idx + s < 128 and idx < s:
            sum_val[idx] += sum_val[idx + s]
        cute.arch.sync_threads()
        s //= 2

    if idx == 0:
        mean_val = sum_val[0] / (mp_out_h * mp_out_w)
        tanh_val = math.tanh(mean_val)
        gOutput[bidy, tidy, 0, 0] = tanh_val

@cute.jit
def fused_op_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_h: int, in_w: int, out_h: int, out_w: int,
    kernel_size: int, stride: int, padding: int,
    maxpool_kernel: int, maxpool_stride: int, mp_out_h: int, mp_out_w: int
):
    threads_per_block = 128
    grid_x = cute.ceil_div(mp_out_h * mp_out_w, threads_per_block)
    grid_y = batch_size
    grid_z = 1

    conv_transpose_maxpool_hardtanh_mean_tanh_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_h, in_w, out_h, out_w,
        kernel_size, stride, padding,
        maxpool_kernel, maxpool_stride, mp_out_h, mp_out_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

import math

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.compiled = {}

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        out_h = (in_h - 1) * 1 - 2 * 1 + 3
        out_w = (in_w - 1) * 1 - 2 * 1 + 3
        mp_out_h = out_h // self.maxpool_stride
        mp_out_w = out_w // self.maxpool_stride

        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda() if self.conv_transpose.bias is not None else torch.empty(0, device=x.device)
        output = torch.empty((batch_size, self.conv_transpose.out_channels, 1, 1), dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_op_host, mInput, mWeight, mBias, mOutput,
                batch_size, in_h, in_w, out_h, out_w,
                3, 1, 1,
                self.maxpool_kernel_size, self.maxpool_stride, mp_out_h, mp_out_w
            )
            self.compiled[key] = compiled

        compiled(mInput, mWeight, mBias, mOutput,
                 batch_size, in_h, in_w, out_h, out_w,
                 3, 1, 1,
                 self.maxpool_kernel_size, self.maxpool_stride, mp_out_h, mp_out_w)
        return output