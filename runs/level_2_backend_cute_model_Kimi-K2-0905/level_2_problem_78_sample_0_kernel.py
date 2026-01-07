import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_transpose_conv_maxpool_sum_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_size: int, stride: int, padding: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    block_idx = bidz * cute.arch.grid_dim().x * cute.arch.grid_dim().y + bidy * cute.arch.grid_dim().x + bidx
    thread_idx = block_idx * bdimx * bdimy * bdimz + tidz * bdimx * bdimy + tidy * bdimx + tidx

    total_threads = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z * bdimx * bdimy * bdimz

    for idx in range(thread_idx, batch_size * out_channels * out_d * out_h * out_w, total_threads):
        n = idx // (out_channels * out_d * out_h * out_w)
        c = (idx // (out_d * out_h * out_w)) % out_channels
        d = (idx // (out_h * out_w)) % out_d
        h = (idx // out_w) % out_h
        w = idx % out_w

        sum_val = 0.0

        # Transposed convolution
        for ic in range(in_channels):
            for kd in range(kernel_size):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        in_d_idx = (d - kd + padding) // stride
                        in_h_idx = (h - kh + padding) // stride
                        in_w_idx = (w - kw + padding) // stride

                        if in_d_idx >= 0 and in_d_idx < in_d and in_h_idx >= 0 and in_h_idx < in_h and in_w_idx >= 0 and in_w_idx < in_w:
                            weight_val = gWeight[ic, c, kd, kh, kw]
                            input_val = gInput[n, ic, in_d_idx, in_h_idx, in_w_idx]
                            sum_val += weight_val * input_val

        sum_val += gBias[c]

        # MaxPool1 (kernel=2, stride=2)
        max1_val = -float('inf')
        for pd in range(2):
            for ph in range(2):
                for pw in range(2):
                    od = d // 2 + pd
                    oh = h // 2 + ph
                    ow = w // 2 + pw
                    if od < out_d and oh < out_h and ow < out_w:
                        max1_val = max(max1_val, sum_val)

        # MaxPool2 (kernel=3, stride=3)
        max2_val = -float('inf')
        for pd in range(3):
            for ph in range(3):
                for pw in range(3):
                    od = d // 3 + pd
                    oh = h // 3 + ph
                    ow = w // 3 + pw
                    if od < out_d and oh < out_h and ow < out_w:
                        max2_val = max(max2_val, max1_val)

        # Sum reduction over channels
        if c == 0:
            channel_sum = 0.0
            for oc in range(out_channels):
                channel_sum += max2_val
            gOutput[n, 0, d, h, w] = channel_sum
        cute.arch.sync_threads()

@cute.jit
def fused_transpose_conv_maxpool_sum_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_size: int, stride: int, padding: int
):
    threads_per_block = 256
    total_elements = batch_size * out_channels * out_d * out_h * out_w
    grid_x = cute.ceil_div(total_elements, threads_per_block)

    fused_transpose_conv_maxpool_sum_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_size, stride, padding
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        out_d = (in_d - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_h = (in_h - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_w = (in_w - 1) * self.stride - 2 * self.padding + self.kernel_size

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        output = torch.empty(batch_size, 1, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_transpose_conv_maxpool_sum_host,
                mInput, mWeight, mBias, mOutput,
                batch_size, self.in_channels, self.out_channels,
                in_d, in_h, in_w, out_d, out_h, out_w,
                self.kernel_size, self.stride, self.padding
            )
            self.compiled[key] = compiled

        compiled(
            mInput, mWeight, mBias, mOutput,
            batch_size, self.in_channels, self.out_channels,
            in_d, in_h, in_w, out_d, out_h, out_w,
            self.kernel_size, self.stride, self.padding
        )
        return output