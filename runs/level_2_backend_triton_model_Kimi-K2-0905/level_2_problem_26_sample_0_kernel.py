import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_add_hardswish_kernel(
    x_ptr,      # pointer to transposed conv output
    add_ptr,    # pointer to add tensor
    out_ptr,    # pointer to output
    B, C, D, H, W,
    stride_d, stride_h, stride_w,
    stride_bx, stride_cx, stride_dx, stride_hx, stride_wx,
    stride_ba, stride_ca, stride_da, stride_ha, stride_wa,
    stride_bo, stride_co, stride_do, stride_ho, stride_wo,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    c_start = pid_c * BLOCK_C
    dhw_start = pid_dhw * BLOCK_DHW

    cs = c_start + tl.arange(0, BLOCK_C)
    dhw = dhw_start + tl.arange(0, BLOCK_DHW)

    mask_c = cs < C
    mask_dhw = dhw < D * H * W

    # Compute 3D indices from linear dhw
    d = dhw // (H * W)
    hw = dhw % (H * W)
    h = hw // W
    w = hw % W

    # Compute memory offsets
    off_x = (pid_b * stride_bx + cs[:, None] * stride_cx +
             d[None, :] * stride_dx + h[None, :] * stride_hx + w[None, :] * stride_wx)
    off_add = (pid_b * stride_ba + cs[:, None] * stride_ca +
               d[None, :] * stride_da + h[None, :] * stride_ha + w[None, :] * stride_wa)
    off_out = (pid_b * stride_bo + cs[:, None] * stride_co +
               d[None, :] * stride_do + h[None, :] * stride_ho + w[None, :] * stride_wo)

    mask = mask_c[:, None] & mask_dhw[None, :]

    x = tl.load(x_ptr + off_x, mask=mask, other=0.0)
    add_val = tl.load(add_ptr + off_add, mask=mask, other=0.0)

    x = x + add_val
    # HardSwish: x * hardsigmoid(x) = x * relu6(x+3)/6
    x_plus3 = x + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus3, 0.0), 6.0)
    hardsigmoid = relu6 / 6.0
    out = x * hardsigmoid

    tl.store(out_ptr + off_out, out, mask=mask)


def triton_fused_transpose_add_hardswish(conv_out, add_tensor):
    assert conv_out.is_cuda and add_tensor.is_cuda
    B, C, D, H, W = conv_out.shape
    out = torch.empty_like(conv_out)

    # Get strides
    stride_bx, stride_cx, stride_dx, stride_hx, stride_wx = conv_out.stride()
    stride_ba, stride_ca, stride_da, stride_ha, stride_wa = add_tensor.stride()
    stride_bo, stride_co, stride_do, stride_ho, stride_wo = out.stride()

    BLOCK_C = 16
    BLOCK_DHW = 64

    grid = (B, (C + BLOCK_C - 1) // BLOCK_C, (D * H * W + BLOCK_DHW - 1) // BLOCK_DHW)

    fused_transpose_add_hardswish_kernel[grid](
        conv_out, add_tensor, out,
        B, C, D, H, W,
        1, 1, 1,
        stride_bx, stride_cx, stride_dx, stride_hx, stride_wx,
        stride_ba, stride_ca, stride_da, stride_ha, stride_wa,
        stride_bo, stride_co, stride_do, stride_ho, stride_wo,
        BLOCK_C=BLOCK_C, BLOCK_DHW=BLOCK_DHW
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = triton_fused_transpose_add_hardswish(x, add_input)
        return x