import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_clamp_div_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_c, in_d, in_h, in_w,
    out_c, out_d, out_h, out_w,
    k_d, k_h, k_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
    min_val, divisor,
    BLOCK_C: tl.constexpr, BLOCK_DHW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    # Compute output spatial indices
    hw_per_block = BLOCK_DHW
    out_d = pid_dhw // (out_h * out_w)
    rem = pid_dhw % (out_h * out_w)
    out_h_idx = rem // out_w
    out_w_idx = rem % out_w

    # Compute input spatial start indices
    in_d_start = out_d * stride_d - pad_d
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w

    # Accumulate over input channels and kernel volume
    acc = 0.0
    for ic in range(0, in_c, BLOCK_C):
        ic_off = tl.arange(0, BLOCK_C)
        ic_mask = ic_off < (in_c - ic)

        for kd in range(k_d):
            for kh in range(k_h):
                for kw in range(k_w):
                    in_d_idx = in_d_start + kd
                    in_h_idx = in_h_start + kh
                    in_w_idx = in_w_start + kw

                    # Bounds check for input
                    in_bounds = (in_d_idx >= 0) & (in_d_idx < in_d) & \
                                (in_h_idx >= 0) & (in_h_idx < in_h) & \
                                (in_w_idx >= 0) & (in_w_idx < in_w)

                    # Input pointer
                    x_offset = (
                        pid_b * in_c * in_d * in_h * in_w +
                        (ic + ic_off) * in_d * in_h * in_w +
                        in_d_idx * in_h * in_w +
                        in_h_idx * in_w +
                        in_w_idx
                    )

                    # Weight pointer
                    w_offset = (
                        pid_c * in_c * k_d * k_h * k_w +
                        (ic + ic_off) * k_d * k_h * k_w +
                        kd * k_h * k_w +
                        kh * k_w +
                        kw
                    )

                    x_val = tl.load(x_ptr + x_offset, mask=ic_mask & in_bounds, other=0.0)
                    w_val = tl.load(w_ptr + w_offset, mask=ic_mask, other=0.0)
                    acc += tl.sum(x_val * w_val)

    # Add bias
    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_c)
        acc += b_val

    # Clamp and divide
    acc = tl.maximum(acc, min_val)
    acc = acc / divisor

    # Store output
    out_offset = (
        pid_b * out_c * out_d * out_h * out_w +
        pid_c * out_d * out_h * out_w +
        pid_dhw
    )
    tl.store(out_ptr + out_offset, acc)


def triton_conv_transpose3d_clamp_div(x, weight, bias, stride, padding, min_val, divisor):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_c, in_d, in_h, in_w = x.shape
    out_c, _, k_d, k_h, k_w = weight.shape
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    out_d = (in_d - 1) * stride_d - 2 * pad_d + k_d
    out_h = (in_h - 1) * stride_h - 2 * pad_h + k_h
    out_w = (in_w - 1) * stride_w - 2 * pad_w + k_w

    out = torch.empty(batch_size, out_c, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

    BLOCK_C = 16
    BLOCK_DHW = 64

    grid = (batch_size, out_c, out_d * out_h * out_w)

    conv_transpose3d_clamp_div_kernel[grid](
        x, weight, bias, out,
        batch_size, in_c, in_d, in_h, in_w,
        out_c, out_d, out_h, out_w,
        k_d, k_h, k_w, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        min_val, divisor,
        BLOCK_C=BLOCK_C, BLOCK_DHW=BLOCK_DHW
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = triton_conv_transpose3d_clamp_div(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.conv_transpose.stride, self.conv_transpose.padding,
            self.min_value, self.divisor
        )
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]