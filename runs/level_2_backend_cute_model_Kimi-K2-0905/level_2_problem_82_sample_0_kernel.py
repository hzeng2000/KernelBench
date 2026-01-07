import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv_tanh_scale_bias_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    scale: float, N: int, H: int, W: int, C_out: int, C_in: int,
    KH: int, KW: int, pad_h: int, pad_w: int, stride_h: int, stride_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    hw = bidx * bdimx + tidx
    c_out = bidy * bdimy + tidy
    n = bidz * bdimz + tidz

    if n >= N or c_out >= C_out:
        return

    out_h = (H + 2 * pad_h - KH) // stride_h + 1
    out_w = (W + 2 * pad_w - KW) // stride_w + 1

    if hw >= out_h * out_w:
        return

    oh = hw // out_w
    ow = hw % out_w

    acc = 0.0
    for c_in in range(C_in):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh * stride_h - pad_h + kh
                iw = ow * stride_w - pad_w + kw
                if 0 <= ih < H and 0 <= iw < W:
                    x_val = gX[n, c_in, ih, iw]
                    w_val = gW[c_out, c_in, kh, kw]
                    acc += x_val * w_val

    acc += gB[c_out, 0, 0]
    acc = cute.math.tanh(acc)
    acc = acc * scale
    gY[n, c_out, oh, ow] = acc

@cute.kernel
def maxpool_kernel(
    gX: cute.Tensor, gY: cute.Tensor,
    N: int, C: int, H: int, W: int,
    pool_h: int, pool_w: int, stride_h: int, stride_w: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    hw = bidx * bdimx + tidx
    c = bidy * bdimy + tidy
    n = bidz * bdimz + tidz

    if n >= N or c >= C:
        return

    out_h = H // stride_h
    out_w = W // stride_w

    if hw >= out_h * out_w:
        return

    oh = hw // out_w
    ow = hw % out_w

    max_val = -float('inf')
    for ph in range(pool_h):
        for pw in range(pool_w):
            ih = oh * stride_h + ph
            iw = ow * stride_w + pw
            if ih < H and iw < W:
                val = gX[n, c, ih, iw]
                if val > max_val:
                    max_val = val

    gY[n, c, oh, ow] = max_val

@cute.jit
def conv_tanh_scale_bias_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    scale: float, N: int, H: int, W: int, C_out: int, C_in: int,
    KH: int, KW: int, pad_h: int, pad_w: int, stride_h: int, stride_w: int
):
    out_h = (H + 2 * pad_h - KH) // stride_h + 1
    out_w = (W + 2 * pad_w - KW) // stride_w + 1
    total_hw = out_h * out_w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_hw, threads_per_block)
    grid_y = C_out
    grid_z = N
    conv_tanh_scale_bias_kernel(
        mX, mW, mB, mY, scale, N, H, W, C_out, C_in, KH, KW, pad_h, pad_w, stride_h, stride_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

@cute.jit
def maxpool_host(
    mX: cute.Tensor, mY: cute.Tensor,
    N: int, C: int, H: int, W: int,
    pool_h: int, pool_w: int, stride_h: int, stride_w: int
):
    out_h = H // stride_h
    out_w = W // stride_w
    total_hw = out_h * out_w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_hw, threads_per_block)
    grid_y = C
    grid_z = N
    maxpool_kernel(
        mX, mY, N, C, H, W, pool_h, pool_w, stride_h, stride_w
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.compiled = {}

    def forward(self, x):
        N, C_in, H, W = x.shape
        C_out = self.conv_weight.shape[0]
        K = self.kernel_size
        pad = K // 2
        stride = 1
        out_h = (H + 2 * pad - K) // stride + 1
        out_w = (W + 2 * pad - K) // stride + 1

        x = x.contiguous().cuda()
        conv_out = torch.empty((N, C_out, out_h, out_w), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(self.conv_weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(self.bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mC = from_dlpack(conv_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv_tanh_scale_bias_host,
                mX, mW, mB, mC,
                self.scaling_factor, N, H, W, C_out, C_in,
                K, K, pad, pad, stride, stride
            )
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mC, self.scaling_factor, N, H, W, C_out, C_in, K, K, pad, pad, stride, stride)

        pool_k = self.pool_kernel_size
        pool_stride = pool_k
        pool_out_h = out_h // pool_stride
        pool_out_w = out_w // pool_stride
        pool_out = torch.empty((N, C_out, pool_out_h, pool_out_w), dtype=x.dtype, device=x.device)

        mP = from_dlpack(pool_out, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        compiled_pool = self.compiled.get((x.dtype, 'pool'))
        if compiled_pool is None:
            compiled_pool = cute.compile(
                maxpool_host,
                mC, mP, N, C_out, out_h, out_w, pool_k, pool_k, pool_stride, pool_stride
            )
            self.compiled[(x.dtype, 'pool')] = compiled_pool

        compiled_pool(mC, mP, N, C_out, out_h, out_w, pool_k, pool_k, pool_stride, pool_stride)
        return pool_out