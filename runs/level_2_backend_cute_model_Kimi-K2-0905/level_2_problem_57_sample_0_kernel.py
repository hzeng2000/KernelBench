import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_relu_hardswish_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    N: int, H: int, W: int, C_in: int, C_out: int, K: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_idx = bidx * bdimx + tidx
    total_threads = bidz * bdimz * bdimy * bdimx

    out_hw = H * W
    out_size = N * out_hw * C_out

    for idx in range(thread_idx, out_size, total_threads):
        n = idx // (out_hw * C_out)
        rem = idx % (out_hw * C_out)
        c_out = rem // out_hw
        hw = rem % out_hw
        h_out = hw // W
        w_out = hw % W

        sum_val = 0.0
        for c_in in range(C_in):
            for kh in range(K):
                for kw in range(K):
                    h_in = h_out + kh
                    w_in = w_out + kw
                    if h_in < H and w_in < W:
                        x_idx = n * (C_in * H * W) + c_in * (H * W) + h_in * W + w_in
                        w_idx = c_out * (C_in * K * K) + c_in * (K * K) + kh * K + kw
                        sum_val += gX[x_idx] * gW[w_idx]

        if gB:
            sum_val += gB[c_out]

        relu_val = max(sum_val, 0.0)
        hardswish_val = relu_val * min(max((relu_val + 3.0) / 6.0, 0.0), 1.0)

        y_idx = n * (C_out * H * W) + c_out * (H * W) + h_out * W + w_out
        gY[y_idx] = hardswish_val

@cute.jit
def conv_relu_hardswish_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    N: int, H: int, W: int, C_in: int, C_out: int, K: int
):
    total_threads = 256
    total_elems = N * H * W * C_out
    grid_x = cute.ceil_div(total_elems, total_threads)

    conv_relu_hardswish_kernel(mX, mW, mB, mY, N, H, W, C_in, C_out, K).launch(
        grid=(grid_x, 1, 1), block=(total_threads, 1, 1)
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.compiled = {}

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.kernel_size
        C_out = self.out_channels
        C_in = self.in_channels

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()

        out_h = H
        out_w = W
        y = torch.empty(N, C_out, out_h, out_w, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x.view(-1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mW = from_dlpack(weight.view(-1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mB = from_dlpack(bias.view(-1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(y.view(-1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_relu_hardswish_host, mX, mW, mB, mY,
                N, H, W, C_in, C_out, K
            )
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mY, N, H, W, C_in, C_out, K)
        return y