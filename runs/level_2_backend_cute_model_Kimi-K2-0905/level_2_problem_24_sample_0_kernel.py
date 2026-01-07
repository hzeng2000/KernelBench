import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def conv3d_kernel(
    gInput: cute.Tensor, gWeight: cute.Tensor, gBias: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    D: int, H: int, W: int, kernel_size: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_idx = (bidz * cute.arch.grid_dim().x * cute.arch.grid_dim().y +
                  bidy * cute.arch.grid_dim().x + bidx) * (bdimx * bdimy * bdimz) +
                 tidz * bdimx * bdimy + tidy * bdimx + tidx

    out_d = (H - kernel_size + 1)
    out_h = (H - kernel_size + 1)
    out_w = (W - kernel_size + 1)
    total_out = batch_size * out_channels * out_d * out_h * out_w

    if thread_idx >= total_out:
        return

    # Compute output indices
    tmp = thread_idx
    out_w_idx = tmp % out_w
    tmp //= out_w
    out_h_idx = tmp % out_h
    tmp //= out_h
    out_d_idx = tmp % out_d
    tmp //= out_d
    out_c_idx = tmp % out_channels
    tmp //= out_channels
    b_idx = tmp

    # Compute convolution
    sum_val = 0.0
    for kc in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_d = out_d_idx + kd
                    in_h = out_h_idx + kh
                    in_w = out_w_idx + kw
                    if in_d < D and in_h < H and in_w < W:
                        in_val = gInput[b_idx, kc, in_d, in_h, in_w]
                        weight_val = gWeight[out_c_idx, kc, kd, kh, kw]
                        sum_val += in_val * weight_val
    gOutput[b_idx, out_c_idx, out_d_idx, out_h_idx, out_w_idx] = sum_val

@cute.kernel
def min_softmax_kernel(
    gInput: cute.Tensor, gOutput: cute.Tensor,
    batch_size: int, out_channels: int, out_d: int, out_h: int, out_w: int, dim: int
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    thread_idx = (bidz * cute.arch.grid_dim().x * cute.arch.grid_dim().y +
                  bidy * cute.arch.grid_dim().x + bidx) * (bdimx * bdimy * bdimz) +
                 tidz * bdimx * bdimy + tidy * bdimx + tidx

    total_threads = batch_size * out_channels * out_h * out_w
    if thread_idx >= total_threads:
        return

    # Compute indices for min reduction and softmax
    tmp = thread_idx
    w_idx = tmp % out_w
    tmp //= out_w
    h_idx = tmp % out_h
    tmp //= out_h
    c_idx = tmp % out_channels
    tmp //= out_channels
    b_idx = tmp

    # Find min along dim (depth dimension)
    min_val = gInput[b_idx, c_idx, 0, h_idx, w_idx]
    for d in range(1, out_d):
        val = gInput[b_idx, c_idx, d, h_idx, w_idx]
        if val < min_val:
            min_val = val

    # Compute softmax
    exp_sum = 0.0
    for oc in range(out_channels):
        val = gInput[b_idx, oc, 0, h_idx, w_idx] if dim == 2 else min_val
        exp_val = cute.math.exp(val)
        exp_sum += exp_val

    # Write softmax output
    out_idx = b_idx * out_channels * out_h * out_w + c_idx * out_h * out_w + h_idx * out_w + w_idx
    if dim == 2:
        val = gInput[b_idx, c_idx, 0, h_idx, w_idx]
    else:
        val = min_val
    gOutput[out_idx] = cute.math.exp(val) / exp_sum

@cute.jit
def conv3d_min_softmax_host(
    mInput: cute.Tensor, mWeight: cute.Tensor, mBias: cute.Tensor, mOutput: cute.Tensor,
    batch_size: int, in_channels: int, out_channels: int,
    D: int, H: int, W: int, kernel_size: int, dim: int
):
    out_d = D - kernel_size + 1
    out_h = H - kernel_size + 1
    out_w = W - kernel_size + 1

    # Launch conv3d kernel
    total_conv_threads = batch_size * out_channels * out_d * out_h * out_w
    threads_per_block = 256
    grid_x = cute.ceil_div(total_conv_threads, threads_per_block)

    conv3d_kernel(
        mInput, mWeight, mBias, mOutput,
        batch_size, in_channels, out_channels, D, H, W, kernel_size
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

    # Launch min+softmax kernel
    total_min_softmax_threads = batch_size * out_channels * out_h * out_w
    grid_x = cute.ceil_div(total_min_softmax_threads, threads_per_block)

    min_softmax_kernel(
        mOutput, mOutput,
        batch_size, out_channels, out_d, out_h, out_w, dim
    ).launch(grid=(grid_x, 1, 1), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.kernel_size = kernel_size
        self.dim = dim
        self.compiled = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, D, H, W = x.shape
        out_channels = self.weight.shape[0]
        out_d = D - self.kernel_size + 1
        out_h = H - self.kernel_size + 1
        out_w = W - self.kernel_size + 1

        x = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias = self.bias.contiguous().cuda()

        # Intermediate output for conv3d
        conv_output = torch.empty((batch_size, out_channels, out_d, out_h, out_w), dtype=x.dtype, device=x.device)
        # Final output after min+softmax
        final_output = torch.empty((batch_size, out_channels, out_h, out_w), dtype=x.dtype, device=x.device)

        mInput = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mWeight = from_dlpack(weight, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mConvOutput = from_dlpack(conv_output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mFinalOutput = from_dlpack(final_output, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        key = (x.dtype, self.kernel_size, self.dim)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(
                conv3d_min_softmax_host,
                mInput, mWeight, mBias, mConvOutput,
                batch_size, in_channels, out_channels, D, H, W, self.kernel_size, self.dim
            )
            self.compiled[key] = compiled

        compiled(mInput, mWeight, mBias, mConvOutput, batch_size, in_channels, out_channels, D, H, W, self.kernel_size, self.dim)
        return final_output