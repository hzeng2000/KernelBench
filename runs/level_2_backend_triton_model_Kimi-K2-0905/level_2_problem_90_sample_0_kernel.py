import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_leaky_add_clamp_gelu_kernel(
    x_ptr, w_ptr, b_ptr, sum_ptr, out_ptr,
    batch, out_c, out_d, out_h, out_w,
    in_c, in_d, in_h, in_w,
    k_size, stride, pad, negative_slope,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_hw = tl.program_id(3)

    hw = pid_hw * BLOCK_H * BLOCK_W + tl.arange(0, BLOCK_H * BLOCK_W)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    c_off = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    d_mask = d_off < out_d
    hw_mask = hw < (out_h * out_w)
    c_mask = c_off < out_c

    d = d_off[:, None, None, None]
    h = (hw // out_w)[None, :, None, None]
    w = (hw % out_w)[None, None, :, None]
    c = c_off[None, None, None, :]

    in_d_start = d * stride - pad
    in_h_start = h * stride - pad
    in_w_start = w * stride - pad

    acc = tl.zeros([BLOCK_D, BLOCK_H * BLOCK_W, BLOCK_C], dtype=tl.float32)

    for ic in range(in_c):
        for kd in range(k_size):
            for kh in range(k_size):
                for kw in range(k_size):
                    in_d_idx = in_d_start + kd
                    in_h_idx = in_h_start + kh
                    in_w_idx = in_w_start + kw

                    in_bounds = (
                        (in_d_idx >= 0) & (in_d_idx < in_d) &
                        (in_h_idx >= 0) & (in_h_idx < in_h) &
                        (in_w_idx >= 0) & (in_w_idx < in_w)
                    )

                    in_idx = (
                        pid_b * in_c * in_d * in_h * in_w +
                        ic * in_d * in_h * in_w +
                        in_d_idx * in_h * in_w +
                        in_h_idx * in_w +
                        in_w_idx
                    )

                    x_val = tl.load(x_ptr + in_idx, mask=in_bounds, other=0.0)

                    w_idx = (
                        c_off * in_c * k_size * k_size * k_size +
                        ic * k_size * k_size * k_size +
                        kd * k_size * k_size +
                        kh * k_size +
                        kw
                    )

                    w_val = tl.load(w_ptr + w_idx, mask=c_mask, other=0.0)

                    acc += x_val * w_val

    if b_ptr is not None:
        b_val = tl.load(b_ptr + c_off, mask=c_mask, other=0.0)
        acc += b_val

    acc = tl.where(acc > 0, acc, acc * negative_slope)

    sum_idx = c_off[None, None, None, :] * 1 * 1 * 1
    sum_val = tl.load(sum_ptr + sum_idx, mask=c_mask, other=0.0)
    acc += sum_val

    acc = tl.clamp(acc, -1.0, 1.0)

    acc = 0.5 * acc * (1.0 + tl.tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

    out_idx = (
        pid_b * out_c * out_d * out_h * out_w +
        c_off[:, None, None, None] * out_d * out_h * out_w +
        d_off[None, :, None, None] * out_h * out_w +
        h * out_w +
        w
    )

    tl.store(out_ptr + out_idx, acc, mask=d_mask & hw_mask & c_mask[:, None, None, None])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous() if self.conv.bias is not None else None
        sum_t = self.sum_tensor.contiguous()

        batch, in_c, in_d, in_h, in_w = x.shape
        out_c, _, k_size, _, _ = w.shape
        stride = self.conv.stride[0]
        pad = self.conv.padding[0]

        out_d = (in_d + 2 * pad - k_size) // stride + 1
        out_h = (in_h + 2 * pad - k_size) // stride + 1
        out_w = (in_w + 2 * pad - k_size) // stride + 1

        out = torch.empty(batch, out_c, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

        BLOCK_C = 8
        BLOCK_D = 4
        BLOCK_H = 4
        BLOCK_W = 4

        grid = (
            batch,
            (out_c + BLOCK_C - 1) // BLOCK_C,
            (out_d + BLOCK_D - 1) // BLOCK_D,
            (out_h * out_w + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W),
        )

        fused_conv_leaky_add_clamp_gelu_kernel[grid](
            x, w, b, sum_t, out,
            batch, out_c, out_d, out_h, out_w,
            in_c, in_d, in_h, in_w,
            k_size, stride, pad, 0.2,
            BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )

        return out


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]