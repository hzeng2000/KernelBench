import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv3d_mish_tanh_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    D_in: int, H_in: int, W_in: int,
    D_out: int, H_out: int, W_out: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_x >= W_out or out_y >= H_out or out_z >= D_out:
        return

    for n in range(batch_size):
        for oc in range(out_channels):
            acc = 0.0
            for ic in range(in_channels):
                for kd in range(kernel_d):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            in_d = out_z * stride_d - pad_d + kd
                            in_h = out_y * stride_h - pad_h + kh
                            in_w = out_x * stride_w - pad_w + kw
                            if 0 <= in_d < D_in and 0 <= in_h < H_in and 0 <= in_w < W_in:
                                acc += gInput[n, ic, in_d, in_h, in_w] * gWeight[oc, ic, kd, kh, kw]
            if gBias.shape[0] > 0:
                acc += gBias[oc]
            # Mish activation
            mish_val = acc * cute.math.tanh(cute.math.soft_relu(acc))
            # Tanh activation
            tanh_val = cute.math.tanh(mish_val)
            gOutput[n, oc, out_z, out_y, out_x] = tanh_val

@cute.jit
def conv3d_mish_tanh_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    D_in: int, H_in: int, W_in: int,
    D_out: int, H_out: int, W_out: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(W_out, threads_per_block)
    grid_y = cute.ceil_div(H_out, threads_per_block)
    grid_z = cute.ceil_div(D_out, threads_per_block)

    conv3d_mish_tanh_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_channels, out_channels,
        D_in, H_in, W_in, D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.compiled = {}

    def forward(self, x):
        batch_size, in_channels, D_in, H_in, W_in = x.shape
        D_out = (D_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        H_out = (H_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        W_out = (W_in + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1

        x = x.contiguous().cuda()
        output = torch.empty(batch_size, self.out_channels, D_out, H_out, W_out, dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv3d_mish_tanh_host,
                mInput, mWeight, mBias, mOutput,
                batch_size, in_channels, self.out_channels,
                D_in, H_in, W_in, D_out, H_out, W_out,
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.stride[0], self.stride[1], self.stride[2],
                self.padding[0], self.padding[1], self.padding[2]
            )
            self.compiled[key] = compiled

        compiled(
            mInput, mWeight, mBias, mOutput,
            batch_size, in_channels, self.out_channels,
            D_in, H_in, W_in, D_out, H_out, W_out,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2]
        )
        return output