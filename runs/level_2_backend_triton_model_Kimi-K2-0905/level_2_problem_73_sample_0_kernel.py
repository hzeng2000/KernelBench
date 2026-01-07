import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_bn_scale_kernel(
    x_ptr, w_ptr, b_ptr, running_mean_ptr, running_var_ptr, out_ptr,
    batch, in_c, out_c, h, w, k,
    stride, padding, dilation, groups,
    eps, scale,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw = h * w
    out_h = (h + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    out_w = (w + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    out_hw = out_h * out_w

    if pid_b >= batch or pid_out_c >= out_c or pid_hw >= out_hw:
        return

    out_y = pid_hw // out_w
    out_x = pid_hw % out_w

    in_y_origin = out_y * stride - padding
    in_x_origin = out_x * stride - padding

    acc = 0.0
    for in_c_idx in range(in_c):
        for kh in range(k):
            for kw in range(k):
                in_y = in_y_origin + kh * dilation
                in_x = in_x_origin + kw * dilation
                if 0 <= in_y < h and 0 <= in_x < w:
                    x_idx = pid_b * in_c * h * w + in_c_idx * h * w + in_y * w + in_x
                    w_idx = pid_out_c * in_c * k * k + in_c_idx * k * k + kh * k + kw
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    b_val = tl.load(b_ptr + pid_out_c)
    acc += b_val

    mean = tl.load(running_mean_ptr + pid_out_c)
    var = tl.load(running_var_ptr + pid_out_c)
    inv_std = tl.rsqrt(var + eps)
    acc = (acc - mean) * inv_std

    acc = acc * scale

    out_idx = pid_b * out_c * out_h * out_w + pid_out_c * out_h * out_w + pid_hw
    tl.store(out_ptr + out_idx, acc)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        batch, in_c, h, w = x.shape
        out_c = self.conv.out_channels
        k = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding = self.conv.padding[0]
        dilation = self.conv.dilation[0]
        groups = self.conv.groups
        eps = self.bn.eps

        out_h = (h + 2 * padding - dilation * (k - 1) - 1) // stride + 1
        out_w = (w + 2 * padding - dilation * (k - 1) - 1) // stride + 1

        out = torch.empty(batch, out_c, out_h, out_w, device=x.device, dtype=x.dtype)

        grid = (batch, out_c, out_h * out_w)

        conv_bn_scale_kernel[grid](
            x, self.conv.weight, self.conv.bias,
            self.bn.running_mean, self.bn.running_var, out,
            batch, in_c, out_c, h, w, k,
            stride, padding, dilation, groups,
            eps, self.scaling_factor,
            BLOCK_H=1, BLOCK_W=1
        )

        return out