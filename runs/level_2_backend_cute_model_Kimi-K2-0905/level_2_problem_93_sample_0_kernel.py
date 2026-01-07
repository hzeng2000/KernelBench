import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_conv_transpose_add_clamp_gelu_mul_kernel(
    gInput: cute.Tensor,
    gWeight: cute.Tensor,
    gBias: cute.Tensor,
    gOutput: cute.Tensor,
    add_value: float,
    multiply_value: float,
    N: int, H_out: int, W_out: int, C_out: int,
    stride: int, kernel_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()
    gdimx, gdimy, gdimz = cute.arch.grid_dim()

    hw = bidx * bdimx + tidx
    c = bidy * bdimy + tidy
    n = bidz * bdimz + tidz

    if n < N and c < C_out and hw < (H_out * W_out):
        h_out = hw // W_out
        w_out = hw % W_out

        acc = 0.0
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                h_in = h_out * stride - kh
                w_in = w_out * stride - kw
                if h_in >= 0 and w_in >= 0 and h_in < (H_out // stride) and w_in < (W_out // stride):
                    for ci in range(gInput.shape[1]):
                        acc += gInput[n, ci, h_in, w_in] * gWeight[c, ci, kh, kw]
        acc += gBias[c]

        acc += add_value
        acc = cute.min(acc, 0.0)
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = acc
        x3 = x * x * x
        tanh_arg = 0.7978845608 * (x + 0.044715 * x3)
        tanh_val = cute.tanh(tanh_arg)
        gelu_val = 0.5 * x * (1.0 + tanh_val)
        acc = gelu_val * multiply_value

        gOutput[n, c, h_out, w_out] = acc

@cute.jit
def fused_conv_transpose_add_clamp_gelu_mul_host(
    mInput: cute.Tensor,
    mWeight: cute.Tensor,
    mBias: cute.Tensor,
    mOutput: cute.Tensor,
    add_value: float,
    multiply_value: float,
    stride: int, kernel_size: int
):
    N = mInput.shape[0]
    C_out = mWeight.shape[0]
    H_out = mOutput.shape[2]
    W_out = mOutput.shape[3]

    threads_per_block = 8
    blocks_x = cute.ceil_div(H_out * W_out, threads_per_block)
    blocks_y = cute.ceil_div(C_out, threads_per_block)
    blocks_z = cute.ceil_div(N, threads_per_block)

    fused_conv_transpose_add_clamp_gelu_mul_kernel(
        mInput, mWeight, mBias, mOutput,
        add_value, multiply_value,
        N, H_out, W_out, C_out,
        stride, kernel_size
    ).launch(grid=(blocks_x, blocks_y, blocks_z), block=(threads_per_block, threads_per_block, threads_per_block))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_value = add_value
        self.multiply_value = multiply_value

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(out_channels))

        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H_in, W_in = x.shape
        H_out = H_in * self.stride
        W_out = W_in * self.stride
        x = x.contiguous().cuda()
        output = torch.empty((N, self.out_channels, H_out, W_out), dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mWeight = from_dlpack(self.weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mBias = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mOutput = from_dlpack(output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_conv_transpose_add_clamp_gelu_mul_host,
                mInput, mWeight, mBias, mOutput,
                self.add_value, self.multiply_value,
                self.stride, self.kernel_size
            )
            self.compiled[key] = compiled

        compiled(mInput, mWeight, mBias, mOutput, self.add_value, self.multiply_value, self.stride, self.kernel_size)
        return output