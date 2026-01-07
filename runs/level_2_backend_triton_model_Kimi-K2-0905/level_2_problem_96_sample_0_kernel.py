import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_scale_maxpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    scale, maxpool_kernel, clamp_min, clamp_max,
    BLOCK_C: tl.constexpr, BLOCK_DH: tl.constexpr, BLOCK_DW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1) * BLOCK_C
    pid_od = tl.program_id(2) * BLOCK_DH
    pid_oh = tl.program_id(3) * BLOCK_DH
    pid_ow = tl.program_id(4) * BLOCK_DW

    oc_range = pid_oc + tl.arange(0, BLOCK_C)
    od_range = pid_od + tl.arange(0, BLOCK_DH)
    oh_range = pid_oh + tl.arange(0, BLOCK_DH)
    ow_range = pid_ow + tl.arange(0, BLOCK_DW)

    mask_c = oc_range < out_channels
    mask_d = od_range < out_d
    mask_h = oh_range < out_h
    mask_w = ow_range < out_w

    od, oh, ow = tl.meshgrid(od_range, oh_range, ow_range, indexing='ij')
    mask_dhw = (od < out_d)[:, :, None] & (oh < out_h)[:, None, :] & (ow < out_w)[None, :, :]

    # Compute conv_transpose output
    accum = tl.zeros((BLOCK_C, BLOCK_DH, BLOCK_DW), dtype=tl.float32)
    for ic in range(in_channels):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    id_in = od * stride_d - pad_d + kd
                    ih_in = oh * stride_h - pad_h + kh
                    iw_in = ow * stride_w - pad_w + kw

                    mask_in = (id_in >= 0) & (id_in < in_d) & (ih_in >= 0) & (ih_in < in_h) & (iw_in >= 0) & (iw_in < in_w)
                    x_idx = pid_b * in_channels * in_d * in_h * in_w + ic * in_d * in_h * in_w + id_in * in_h * in_w + ih_in * in_w + iw_in
                    w_idx = pid_oc * in_channels * kernel_d * kernel_h * kernel_w + ic * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw

                    x_val = tl.load(x_ptr + x_idx, mask=mask_in & mask_dhw, other=0.0)
                    w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)
                    accum += x_val * w_val

    # Add bias
    b_val = tl.load(b_ptr + oc_range, mask=mask_c, other=0.0)
    accum = accum + b_val[:, None, None]

    # Scale
    accum = accum * scale

    # MaxPool
    max_val = tl.full((BLOCK_C, BLOCK_DH, BLOCK_DW), float('-inf'), dtype=tl.float32)
    for pd in range(maxpool_kernel):
        for ph in range(maxpool_kernel):
            for pw in range(maxpool_kernel):
                pd_in = od * maxpool_kernel + pd
                ph_in = oh * maxpool_kernel + ph
                pw_in = ow * maxpool_kernel + pw

                mask_pool = (pd_in < out_d) & (ph_in < out_h) & (pw_in < out_w)
                max_val = tl.maximum(max_val, tl.where(mask_pool & mask_dhw, accum, float('-inf')))

    # Global Average Pool (simplified: average across spatial dims)
    global_avg = tl.sum(max_val, axis=(1, 2)) / (out_d * out_h * out_w)
    global_avg = global_avg[:, None, None]

    # Clamp
    clamped = tl.clamp(global_avg, clamp_min, clamp_max)

    # Store output
    out_idx = pid_b * out_channels + oc_range
    tl.store(out_ptr + out_idx, clamped, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        out_channels = self.conv_transpose.out_channels
        kernel_d, kernel_h, kernel_w = self.conv_transpose.kernel_size
        stride_d, stride_h, stride_w = self.conv_transpose.stride
        pad_d, pad_h, pad_w = self.conv_transpose.padding

        out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d
        out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h
        out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w

        w = self.conv_transpose.weight
        b = self.conv_transpose.bias

        out = torch.empty(batch_size, out_channels, 1, 1, 1, device=x.device, dtype=x.dtype)

        BLOCK_C = 16
        BLOCK_DH = 4
        BLOCK_DW = 4

        grid = lambda meta: (
            batch_size,
            (out_channels + BLOCK_C - 1) // BLOCK_C,
            (out_d + BLOCK_DH - 1) // BLOCK_DH,
            (out_h + BLOCK_DH - 1) // BLOCK_DH,
            (out_w + BLOCK_DW - 1) // BLOCK_DW,
        )

        fused_transpose_scale_maxpool_kernel[grid](
            x, w, b, out,
            batch_size, in_channels, out_channels,
            in_d, in_h, in_w,
            out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            self.scale, self.maxpool_kernel_size,
            self.clamp_min, self.clamp_max,
            BLOCK_C=BLOCK_C, BLOCK_DH=BLOCK_DH, BLOCK_DW=BLOCK_DW
        )

        return out.view(batch_size, out_channels)