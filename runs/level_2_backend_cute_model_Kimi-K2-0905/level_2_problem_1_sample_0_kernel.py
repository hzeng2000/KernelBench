import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_relu_bias_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    N: int, H: int, W: int, C_in: int, C_out: int, K: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_id = tidz * bdimx * bdimy + tidy * bdimx + tidx
    block_id = bidz * cute.arch.grid_dim().x * cute.arch.grid_dim().y + bidy * cute.arch.grid_dim().x + bidx

    out_h = H - K + 1
    out_w = W - K + 1

    total_out_pixels = N * out_h * out_w
    pixels_per_thread = cute.ceil_div(total_out_pixels, cute.arch.block_dim().x * cute.arch.grid_dim().x)
    start_pixel = block_id * pixels_per_thread * bdimx + tidx
    end_pixel = min(start_pixel + pixels_per_thread * bdimx, total_out_pixels)

    for p in range(start_pixel, end_pixel, bdimx):
        if p < total_out_pixels:
            n = p // (out_h * out_w)
            rem = p % (out_h * out_w)
            oh = rem // out_w
            ow = rem % out_w

            for c_out in range(C_out):
                acc = 0.0
                for c_in in range(C_in):
                    for kh in range(K):
                        for kw in range(K):
                            ih = oh + kh
                            iw = ow + kw
                            x_val = gX[n, ih, iw, c_in]
                            w_val = gW[c_out, c_in, kh, kw]
                            acc += x_val * w_val
                acc += gB[c_out, 0, 0]
                acc = max(acc, 0.0)
                gY[n, oh, ow, c_out] = acc

@cute.jit
def conv_relu_bias_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    N: int, H: int, W: int, C_in: int, C_out: int, K: int
):
    threads_per_block = 256
    blocks = 128
    conv_relu_bias_kernel(mX, mW, mB, mY, N, H, W, C_in, C_out, K).launch(
        grid=(blocks, 1, 1), block=(threads_per_block, 1, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.compiled = {}

    def forward(self, x):
        N, C_in, H, W = x.shape
        K = self.conv.kernel_size[0]
        C_out = self.conv.out_channels

        x = x.contiguous().cuda()
        W = self.conv.weight.data.contiguous().cuda()
        B = self.bias.data.contiguous().cuda()
        Y = torch.empty((N, C_out, H - K + 1, W - K + 1), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x.permute(0, 2, 3, 1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mY = from_dlpack(Y.permute(0, 2, 3, 1), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_relu_bias_host, mX, mW, mB, mY, N, H, W, C_in, C_out, K)
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mY, N, H, W, C_in, C_out, K)
        return Y