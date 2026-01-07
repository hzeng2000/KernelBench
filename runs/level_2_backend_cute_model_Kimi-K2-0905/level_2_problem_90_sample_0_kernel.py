import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_conv3d_leaky_relu_add_clamp_gelu_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gS: cute.Tensor, gY: cute.Tensor,
    N, C_out, D_out, H_out, W_out, C_in, K, negative_slope
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    global_z = bidz * bdimz + tidz
    global_y = bidy * bdimy + tidy
    global_x = bidx * bdimx + tidx

    if global_x < N and global_y < C_out and global_z < (D_out * H_out * W_out):
        d = global_z // (H_out * W_out)
        h = (global_z // W_out) % H_out
        w = global_z % W_out

        acc = 0.0
        for c in range(C_in):
            for kd in range(K):
                for kh in range(K):
                    for kw in range(K):
                        in_d = d + kd
                        in_h = h + kh
                        in_w = w + kw
                        if in_d < D_out + K - 1 and in_h < H_out + K - 1 and in_w < W_out + K - 1:
                            x_val = gX[global_x, c, in_d, in_h, in_w]
                            w_val = gW[global_y, c, kd, kh, kw]
                            acc += x_val * w_val

        acc += gB[global_y]

        # LeakyReLU
        if acc < 0.0:
            acc *= negative_slope

        # Add sum_tensor
        acc += gS[global_y, 0, 0, 0]

        # Clamp
        acc = max(-1.0, min(1.0, acc))

        # GELU
        acc = 0.5 * acc * (1.0 + tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

        gY[global_x, global_y, d, h, w] = acc

@cute.jit
def fused_conv3d_leaky_relu_add_clamp_gelu_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mS: cute.Tensor, mY: cute.Tensor,
    negative_slope: float
):
    N, C_out, D_out, H_out, W_out = mY.shape
    C_in = mX.shape[1]
    K = mW.shape[2]

    threads_per_block = 256
    grid_x = cute.ceil_div(N, threads_per_block)
    grid_y = cute.ceil_div(C_out, 1)
    grid_z = cute.ceil_div(D_out * H_out * W_out, 1)

    fused_conv3d_leaky_relu_add_clamp_gelu_kernel(
        mX, mW, mB, mS, mY, N, C_out, D_out, H_out, W_out, C_in, K, negative_slope
    ).launch(grid=(grid_x, grid_y, grid_z), block=(threads_per_block, 1, 1))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.compiled = {}

    def forward(self, x):
        N, C_in, D_in, H_in, W_in = x.shape
        K = self.conv.kernel_size[0]
        D_out = D_in - K + 1
        H_out = H_in - K + 1
        W_out = W_in - K + 1
        C_out = self.conv.out_channels

        x = x.contiguous().cuda()
        W = self.conv.weight.contiguous().cuda()
        B = self.conv.bias.contiguous().cuda()
        S = self.sum_tensor.contiguous().cuda()
        Y = torch.empty((N, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)

        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mW = from_dlpack(W, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mS = from_dlpack(S, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mY = from_dlpack(Y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3, 4))

        key = (x.dtype,)
        compiled = self.compiled.get(key)
        if compiled is None:
            compiled = cute.compile(fused_conv3d_leaky_relu_add_clamp_gelu_host, mX, mW, mB, mS, mY, 0.2)
            self.compiled[key] = compiled

        compiled(mX, mW, mB, mS, mY, 0.2)
        return Y