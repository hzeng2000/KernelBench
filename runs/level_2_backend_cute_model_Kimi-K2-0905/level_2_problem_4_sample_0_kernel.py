import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_mish_mish_kernel(
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

    total_threads = cute.arch.grid_dim().x * cute.arch.grid_dim().y * cute.arch.grid_dim().z * bdimx * bdimy * bdimz
    for idx in range(block_id * bdimx * bdimy * bdimz + thread_id, N * out_h * out_w * C_out, total_threads):
        n = idx // (out_h * out_w * C_out)
        rem = idx % (out_h * out_w * C_out)
        c_out = rem // (out_h * out_w)
        rem = rem % (out_h * out_w)
        h_out = rem // out_w
        w_out = rem % out_w

        sum_val = 0.0
        for c_in in range(C_in):
            for kh in range(K):
                for kw in range(K):
                    h_in = h_out + kh
                    w_in = w_out + kw
                    x_val = gX[n, c_in, h_in, w_in]
                    w_val = gW[c_out, c_in, kh, kw]
                    sum_val += x_val * w_val

        if gB:
            sum_val += gB[c_out]

        # Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        x1 = sum_val
        e_x = cute.math.exp(x1)
        softplus = cute.math.log(1.0 + e_x)
        tanh_sp = cute.math.tanh(softplus)
        mish1 = x1 * tanh_sp

        x2 = mish1
        e_x2 = cute.math.exp(x2)
        softplus2 = cute.math.log(1.0 + e_x2)
        tanh_sp2 = cute.math.tanh(softplus2)
        mish2 = x2 * tanh_sp2

        gY[n, c_out, h_out, w_out] = mish2

@cute.jit
def conv_mish_mish_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    N: int, H: int, W: int, C_in: int, C_out: int, K: int
):
    out_h = H - K + 1
    out_w = W - K + 1
    total_elems = N * out_h * out_w * C_out
    threads_per_block = 256
    grid_x = cute.ceil_div(total_elems, threads_per_block)
    conv_mish_mish_kernel(mX, mW, mB, mY, N, H, W, C_in, C_out, K).launch(
        grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1)
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.compiled = {}

    def forward(self, x):
        N, C_in, H, W = x.shape
        K = self.conv.kernel_size[0]
        C_out = self.conv.out_channels

        x = x.contiguous().cuda()
        W = self.conv.weight.data.contiguous().cuda()
        B = self.conv.bias.data.contiguous().cuda() if self.conv.bias is not None else None

        out_h = H - K + 1
        out_w = W - K + 1
        y = torch.empty(N, C_out, out_h, out_w, dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,)) if B is not None else cute.Tensor()
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(conv_mish_mish_host, mX, mW, mB, mY, N, H, W, C_in, C_out, K)
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mY, N, H, W, C_in, C_out, K)
        return y