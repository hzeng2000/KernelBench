import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_conv3d_relu_leaky_gelu_sigmoid_bias_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gOut: cute.Tensor,
    batch: int, in_c: int, out_c: int, d_in: int, h_in: int, w_in: int,
    k: int, d_out: int, h_out: int, w_out: int, negative_slope: float
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    out_x = bidx * bdimx + tidx
    out_y = bidy * bdimy + tidy
    out_z = bidz * bdimz + tidz

    if out_x < w_out and out_y < h_out and out_z < d_out:
        for n in range(batch):
            for oc in range(out_c):
                acc = 0.0
                for ic in range(in_c):
                    for kd in range(k):
                        for kh in range(k):
                            for kw in range(k):
                                in_d = out_z + kd
                                in_h = out_y + kh
                                in_w = out_x + kw
                                if in_d < d_in and in_h < h_in and in_w < w_in:
                                    x_val = gX[n, ic, in_d, in_h, in_w]
                                    w_val = gW[oc, ic, kd, kh, kw]
                                    acc += x_val * w_val
                # Add conv bias
                acc += gB[oc]
                # ReLU
                acc = max(acc, 0.0)
                # LeakyReLU
                acc = acc if acc > 0.0 else acc * negative_slope
                # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x_cubed = acc * acc * acc
                tanh_arg = 0.7978845608 * (acc + 0.044715 * x_cubed)
                # tanh approximation
                tanh_val = tanh_arg / (1.0 + abs(tanh_arg))
                gelu_val = 0.5 * acc * (1.0 + tanh_val)
                # Sigmoid
                sig_val = 1.0 / (1.0 + exp(-gelu_val))
                # Add bias
                final_val = sig_val + gB[oc]
                gOut[n, oc, out_z, out_y, out_x] = final_val

@cute.jit
def fused_conv3d_relu_leaky_gelu_sigmoid_bias_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor,
    batch: int, in_c: int, out_c: int, d_in: int, h_in: int, w_in: int,
    k: int, d_out: int, h_out: int, w_out: int, negative_slope: float
):
    threads_per_block = 8
    grid_x = cute.ceil_div(w_out, threads_per_block)
    grid_y = cute.ceil_div(h_out, threads_per_block)
    grid_z = cute.ceil_div(d_out, threads_per_block)

    fused_conv3d_relu_leaky_gelu_sigmoid_bias_kernel(
        mX, mW, mB, mOut,
        batch, in_c, out_c, d_in, h_in, w_in,
        k, d_out, h_out, w_out, negative_slope
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, threads_per_block, threads_per_block))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        in_c = x.shape[1]
        d_in = x.shape[2]
        h_in = x.shape[3]
        w_in = x.shape[4]
        out_c = self.conv.out_channels
        k = self.conv.kernel_size[0]
        d_out = d_in - k + 1
        h_out = h_in - k + 1
        w_out = w_in - k + 1

        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda().reshape(-1)
        out = torch.empty((batch, out_c, d_out, h_out, w_out), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mB = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOut = from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_conv3d_relu_leaky_gelu_sigmoid_bias_host,
                mX, mW, mB, mOut,
                batch, in_c, out_c, d_in, h_in, w_in,
                k, d_out, h_out, w_out, 0.01
            )
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mOut, batch, in_c, out_c, d_in, h_in, w_in, k, d_out, h_out, w_out, 0.01)

        # Add the learnable bias
        out = out + self.bias
        return out