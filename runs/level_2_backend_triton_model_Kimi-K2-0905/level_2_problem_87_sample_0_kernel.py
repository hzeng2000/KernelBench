import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, out_c, h, w, k,
    stride_h, stride_w, pad_h, pad_w,
    sub1, sub2,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw = pid_hw * BLOCK_H * BLOCK_W + tl.arange(0, BLOCK_H * BLOCK_W)
    mask_hw = hw < h * w
    oh = hw // w
    ow = hw % w

    acc = tl.zeros([BLOCK_H * BLOCK_W], dtype=tl.float32)

    for ic in range(in_c):
        for kh in range(k):
            for kw in range(k):
                ih = oh * stride_h - pad_h + kh
                iw = ow * stride_w - pad_w + kw
                mask_i = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w) & mask_hw
                x_idx = pid_b * in_c * h * w + ic * h * w + ih * w + iw
                x_val = tl.load(x_ptr + x_idx, mask=mask_i, other=0.0)
                w_idx = pid_o * in_c * k * k + ic * k * k + kh * k + kw
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    b_val = tl.load(b_ptr + pid_o)
    out_val = acc + b_val
    out_val = out_val - sub1
    out_val = out_val - sub2

    # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    exp_val = tl.exp(out_val)
    softplus = tl.log(1.0 + exp_val)
    tanh_sp = tl.tanh(softplus)
    mish_out = out_val * tanh_sp

    out_idx = pid_b * out_c * h * w + pid_o * h * w + hw
    tl.store(out_ptr + out_idx, mish_out, mask=mask_hw)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        w = self.conv.weight
        b = self.conv.bias
        batch, in_c, h_in, w_in = x.shape
        out_c = w.shape[0]
        k = w.shape[2]
        stride = self.conv.stride[0]
        pad = self.conv.padding[0]

        h_out = (h_in + 2 * pad - k) // stride + 1
        w_out = (w_in + 2 * pad - k) // stride + 1

        out = torch.empty((batch, out_c, h_out, w_out), dtype=torch.float32, device=x.device)

        BLOCK_H = 8
        BLOCK_W = 8
        grid = (batch, out_c, (h_out * w_out + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W))

        conv_sub_mish_kernel[grid](
            x, w, b, out,
            batch, in_c, out_c, h_out, w_out, k,
            stride, stride, pad, pad,
            self.subtract_value_1, self.subtract_value_2,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
        )
        return out


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]