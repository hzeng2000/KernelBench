import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import math

@cute.kernel
def fused_conv_sub_hardswish_maxpool_mish_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gb: cute.Tensor, gY: cute.Tensor,
    N: int, C_in: int, H_in: int, W_in: int, C_out: int, H_out: int, W_out: int,
    kernel_size: int, subtract_value: float, pool_kernel: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    hw_out = H_out * W_out
    nhw_out = N * hw_out
    hw_in = H_in * W_in

    tid = tidz * bdimx * bdimy + tidy * bdimx + tidx
    bdim = bdimx * bdimy * bdimz
    bid = bidz * cute.arch.grid_dim_x() * cute.arch.grid_dim_y() + bidy * cute.arch.grid_dim_x() + bidx
    global_tid = bid * bdim + tid

    total_threads = cute.arch.grid_dim_x() * cute.arch.grid_dim_y() * cute.arch.grid_dim_z() * bdim

    for idx in range(global_tid, N * C_out * (H_out // pool_kernel) * (W_out // pool_kernel), total_threads):
        out_n = idx // ((C_out * (H_out // pool_kernel) * (W_out // pool_kernel)))
        out_c = (idx // ((H_out // pool_kernel) * (W_out // pool_kernel))) % C_out
        out_h = (idx // (W_out // pool_kernel)) % (H_out // pool_kernel)
        out_w = idx % (W_out // pool_kernel)

        h_start = out_h * pool_kernel
        w_start = out_w * pool_kernel

        max_val = -1e9
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                h_in = h_start + kh - kernel_size // 2
                w_in = w_start + kw - kernel_size // 2

                if h_in >= 0 and h_in < H_in and w_in >= 0 and w_in < W_in:
                    sum_val = 0.0
                    for c_in in range(C_in):
                        x_val = gX[out_n, c_in, h_in, w_in]
                        w_val = gW[out_c, c_in, kh, kw]
                        sum_val += x_val * w_val
                    sum_val += gb[out_c]
                    sum_val -= subtract_value

                    # HardSwish
                    relu6_val = min(max(0.0, sum_val + 3.0), 6.0)
                    hardswish_val = sum_val * relu6_val / 6.0

                    # MaxPool
                    if hardswish_val > max_val:
                        max_val = hardswish_val

        # Mish
        exp_val = math.exp(max_val)
        mish_val = max_val * math.tanh(math.log(1.0 + exp_val))

        gY[out_n, out_c, out_h, out_w] = mish_val

@cute.jit
def fused_conv_sub_hardswish_maxpool_mish_host(
    mX: cute.Tensor, mW: cute.Tensor, mb: cute.Tensor, mY: cute.Tensor,
    N: int, C_in: int, H_in: int, W_in: int, C_out: int, H_out: int, W_out: int,
    kernel_size: int, subtract_value: float, pool_kernel: int
):
    threads_per_block = 256
    total_elems = N * C_out * (H_out // pool_kernel) * (W_out // pool_kernel)
    grid_x = cute.ceil_div(total_elems, threads_per_block)

    fused_conv_sub_hardswish_maxpool_mish_kernel(
        mX, mW, mb, mY, N, C_in, H_in, W_in, C_out, H_out, W_out,
        kernel_size, subtract_value, pool_kernel
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        kernel_size = self.kernel_size
        pool_kernel = self.pool_kernel_size

        H_out = H_in
        W_out = W_in
        H_out_pooled = H_out // pool_kernel
        W_out_pooled = W_out // pool_kernel

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()

        y = torch.empty((N, C_out, H_out_pooled, W_out_pooled), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mb = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype, self.kernel_size, self.pool_kernel_size)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                fused_conv_sub_hardswish_maxpool_mish_host,
                mX, mW, mb, mY, N, C_in, H_in, W_in, C_out, H_out, W_out,
                kernel_size, self.subtract_value, pool_kernel
            )
            self.compiled[key] = compiled

        compiled(mX, mW, mb, mY, N, C_in, H_in, W_in, C_out, H_out, W_out,
                 kernel_size, self.subtract_value, pool_kernel)
        return y