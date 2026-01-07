import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_transpose_scale_pool_bias_scale_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    scale1: float, scale2: float,
    batch_size: int, in_c: int, in_d: int, in_h: int, in_w: int,
    out_c: int, out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_z < out_d and out_y < out_h and out_x < out_w:
        sum_val = 0.0
        for n in range(batch_size):
            for oc in range(out_c):
                acc = 0.0
                for ic in range(in_c):
                    for kd in range(k_d):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                in_d_idx = (out_z + pad_d - kd) // stride_d
                                in_h_idx = (out_y + pad_h - kh) // stride_h
                                in_w_idx = (out_x + pad_w - kw) // stride_w
                                if (out_z + pad_d - kd) % stride_d == 0 and \
                                   (out_y + pad_h - kh) % stride_h == 0 and \
                                   (out_x + pad_w - kw) % stride_w == 0 and \
                                   in_d_idx >= 0 and in_d_idx < in_d and \
                                   in_h_idx >= 0 and in_h_idx < in_h and \
                                   in_w_idx >= 0 and in_w_idx < in_w:
                                    w_val = gWeight[ic, oc, kd, kh, kw]
                                    in_val = gInput[n, ic, in_d_idx, in_h_idx, in_w_idx]
                                    acc += in_val * w_val
                # Apply scale1
                acc *= scale1
                # Average pooling (2x2x2)
                pool_acc = 0.0
                pool_count = 0
                for pd in range(2):
                    for ph in range(2):
                        for pw in range(2):
                            pz = out_z * 2 + pd
                            py = out_y * 2 + ph
                            px = out_x * 2 + pw
                            if pz < out_d and py < out_h and px < out_w:
                                pool_acc += acc
                                pool_count += 1
                if pool_count > 0:
                    pool_acc /= pool_count
                # Add bias and scale2
                biased = pool_acc + gBias[oc, 0, 0, 0]
                scaled = biased * scale2
                # Store to output (pooled dimensions)
                pz = out_z // 2
                py = out_y // 2
                px = out_x // 2
                if pz < out_d // 2 and py < out_h // 2 and px < out_w // 2:
                    gOutput[n, oc, pz, py, px] = scaled

@cute.jit
def conv_transpose_scale_pool_bias_scale_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    scale1: float, scale2: float,
    batch_size: int, in_c: int, in_d: int, in_h: int, in_w: int,
    out_c: int, out_d: int, out_h: int, out_w: int,
    k_d: int, k_h: int, k_w: int, stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int
):
    threads_per_block = 8
    grid_x = cute.ceil_div(out_w, threads_per_block)
    grid_y = cute.ceil_div(out_h, threads_per_block)
    grid_z = cute.ceil_div(out_d, threads_per_block)

    conv_transpose_scale_pool_bias_scale_kernel(
        mInput, mWeight, mBias, mOutput, scale1, scale2,
        batch_size, in_c, in_d, in_h, in_w,
        out_c, out_d, out_h, out_w,
        k_d, k_h, k_w, stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
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
        self.scale1_val = scale1
        self.scale2_val = scale2

        # Initialize weight tensor for transposed conv (in_c, out_c, k_d, k_h, k_w)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_c, in_d, in_h, in_w = x.shape
        # Compute output dimensions
        out_d = (in_d - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        out_h = (in_h - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
        out_w = (in_w - 1) * self.stride[2] - 2 * self.padding[2] + self.kernel_size[2]
        # After avg pool (2x2x2)
        pool_out_d = out_d // 2
        pool_out_h = out_h // 2
        pool_out_w = out_w // 2

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()
        output = torch.empty(batch_size, self.out_channels, pool_out_d, pool_out_h, pool_out_w, dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_transpose_scale_pool_bias_scale_host,
                mInput, mWeight, mBias, mOutput,
                self.scale1_val, self.scale2_val,
                batch_size, in_c, in_d, in_h, in_w,
                self.out_channels, out_d, out_h, out_w,
                self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                self.stride[0], self.stride[1], self.stride[2],
                self.padding[0], self.padding[1], self.padding[2]
            )
            self.compiled[key] = compiled

        compiled(
            mInput, mWeight, mBias, mOutput,
            self.scale1_val, self.scale2_val,
            batch_size, in_c, in_d, in_h, in_w,
            self.out_channels, out_d, out_h, out_w,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2]
        )
        return output