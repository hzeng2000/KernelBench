import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_transpose_conv_residual_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_id = tidz * bdimx * bdimy + tidy * bdimx + tidx
    block_id = bidz * cute.arch.grid_dim_x() * cute.arch.grid_dim_y() + bidy * cute.arch.grid_dim_x() + bidx
    total_threads = bdimx * bdimy * bdimz
    global_thread_id = block_id * total_threads + thread_id

    total_output_elements = batch_size * out_channels * out_d * out_h * out_w
    if global_thread_id >= total_output_elements:
        return

    # Compute output indices
    tmp = global_thread_id
    ow = tmp % out_w
    tmp = tmp // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    tmp = tmp // out_d
    oc = tmp % out_channels
    tmp = tmp // out_channels
    b = tmp % batch_size

    # Compute input region
    in_d_start = od * stride_d - pad_d
    in_h_start = oh * stride_h - pad_h
    in_w_start = ow * stride_w - pad_w

    # Compute convolution
    acc = 0.0
    for ic in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    in_d_idx = in_d_start + kd
                    in_h_idx = in_h_start + kh
                    in_w_idx = in_w_start + kw
                    
                    if (in_d_idx >= 0 and in_d_idx < in_d and
                        in_h_idx >= 0 and in_h_idx < in_h and
                        in_w_idx >= 0 and in_w_idx < in_w):
                        
                        input_val = gInput[b, ic, in_d_idx, in_h_idx, in_w_idx]
                        weight_val = gWeight[oc, ic, kernel_d - 1 - kd, kernel_h - 1 - kh, kernel_w - 1 - kw]
                        acc += input_val * weight_val

    # Add bias
    acc += gBias[oc, 0, 0, 0]

    # Residual operations: x + bias + x + x * x + x = x * (3 + x) + bias
    # But we need to do it step by step as in original
    original = acc
    acc = acc + original  # x + original_x
    acc = acc * original  # (x + original_x) * original_x
    acc = acc + original  # ((x + original_x) * original_x) + original_x

    gOutput[b, oc, od, oh, ow] = acc

@cute.jit
def fused_transpose_conv_residual_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    in_d: int, in_h: int, in_w: int,
    out_d: int, out_h: int, out_w: int,
    kernel_d: int, kernel_h: int, kernel_w: int,
    stride_d: int, stride_h: int, stride_w: int,
    pad_d: int, pad_h: int, pad_w: int,
    out_pad_d: int, out_pad_h: int, out_pad_w: int
):
    total_elements = batch_size * out_channels * out_d * out_h * out_w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elements, threads_per_block)

    fused_transpose_conv_residual_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w, out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        
        # Compute output dimensions
        out_d = (in_d - 1) * self.conv_transpose.stride[0] - 2 * self.conv_transpose.padding[0] + self.conv_transpose.kernel_size[0] + self.conv_transpose.output_padding[0]
        out_h = (in_h - 1) * self.conv_transpose.stride[1] - 2 * self.conv_transpose.padding[1] + self.conv_transpose.kernel_size[1] + self.conv_transpose.output_padding[1]
        out_w = (in_w - 1) * self.conv_transpose.stride[2] - 2 * self.conv_transpose.padding[2] + self.conv_transpose.kernel_size[2] + self.conv_transpose.output_padding[2]
        
        out_channels = self.conv_transpose.out_channels
        output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype, batch_size, in_channels, out_channels, in_d, in_h, in_w, out_d, out_h, out_w)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_transpose_conv_residual_host,
                mInput, mWeight, mBias, mOutput,
                batch_size, in_channels, out_channels,
                in_d, in_h, in_w, out_d, out_h, out_w,
                self.conv_transpose.kernel_size[0], self.conv_transpose.kernel_size[1], self.conv_transpose.kernel_size[2],
                self.conv_transpose.stride[0], self.conv_transpose.stride[1], self.conv_transpose.stride[2],
                self.conv_transpose.padding[0], self.conv_transpose.padding[1], self.conv_transpose.padding[2],
                self.conv_transpose.output_padding[0], self.conv_transpose.output_padding[1], self.conv_transpose.output_padding[2]
            )
            self.compiled[key] = compiled

        compiled(
            mInput, mWeight, mBias, mOutput,
            batch_size, in_channels, out_channels,
            in_d, in_h, in_w, out_d, out_h, out_w,
            self.conv_transpose.kernel_size[0], self.conv_transpose.kernel_size[1], self.conv_transpose.kernel_size[2],
            self.conv_transpose.stride[0], self.conv_transpose.stride[1], self.conv_transpose.stride[2],
            self.conv_transpose.padding[0], self.conv_transpose.padding[1], self.conv_transpose.padding[2],
            self.conv_transpose.output_padding[0], self.conv_transpose.output_padding[1], self.conv_transpose.output_padding[2]
        )
        return output