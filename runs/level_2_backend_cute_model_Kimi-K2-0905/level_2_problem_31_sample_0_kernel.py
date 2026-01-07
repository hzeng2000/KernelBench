import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def fused_conv_min_bias_scale_kernel(
    gX: cute.Tensor, gW: cute.Tensor, gB: cute.Tensor, gY: cute.Tensor,
    N, H, W, C_out, C_in, K, const_val, scale
):
    # Grid-stride loop over output spatial locations and batch
    tid = cute.arch.thread_idx().x + cute.arch.block_idx().x * cute.arch.block_dim().x
    total_threads = cute.arch.grid_dim().x * cute.arch.block_dim().x

    # Output dimensions
    out_H = H - K + 1
    out_W = W - K + 1

    total_out_spatial = N * out_H * out_W * C_out

    for idx in range(tid, total_out_spatial, total_threads):
        n = idx // (out_H * out_W * C_out)
        rem = idx % (out_H * out_W * C_out)
        oh = rem // (out_W * C_out)
        rem = rem % (out_W * C_out)
        ow = rem // C_out
        c_out = rem % C_out

        acc = 0.0
        for c_in in range(C_in):
            for kh in range(K):
                for kw in range(K):
                    ih = oh + kh
                    iw = ow + kw
                    acc += gX[n, ih, iw, c_in] * gW[c_out, kh, kw, c_in]

        # Add bias
        acc += gB[c_out, 0, 0]

        # Min with constant
        acc = cute.min(acc, const_val)

        # Scale
        acc = acc * scale

        gY[n, oh, ow, c_out] = acc

@cute.jit
def fused_conv_min_bias_scale_host(
    mX: cute.Tensor, mW: cute.Tensor, mB: cute.Tensor, mY: cute.Tensor,
    N, H, W, C_out, C_in, K, const_val, scale
):
    threads = 256
    total_out_spatial = N * (H - K + 1) * (W - K + 1) * C_out
    blocks = (total_out_spatial + threads - 1) // threads
    fused_conv_min_bias_scale_kernel(mX, mW, mB, mY, N, H, W, C_out, C_in, K, const_val, scale).launch(
        grid=(blocks, 1, 1), block=(threads, 1, 1)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.compiled = None

    def forward(self, x):
        # Run native conv to get weights
        x_conv = self.conv(x)
        N, C_out, H_out, W_out = x_conv.shape

        # Prepare tensors in NHWC layout for kernel
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous()
        b_nhwc = self.bias.contiguous()
        y_nhwc = torch.empty_like(x_nhwc)

        mX = from_dlpack(x_nhwc, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mW = from_dlpack(w_nhwc, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))
        mB = from_dlpack(b_nhwc, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        mY = from_dlpack(y_nhwc, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3))

        if self.compiled is None:
            self.compiled = cute.compile(
                fused_conv_min_bias_scale_host,
                mX, mW, mB, mY,
                x.shape[0], x.shape[2], x.shape[3], C_out, self.conv.in_channels, self.conv.kernel_size[0],
                self.constant_value, self.scaling_factor
            )

        self.compiled(
            mX, mW, mB, mY,
            x.shape[0], x.shape[2], x.shape[3], C_out, self.conv.in_channels, self.conv.kernel_size[0],
            self.constant_value, self.scaling_factor
        )

        y = y_nhwc.permute(0, 3, 1, 2)
        return y