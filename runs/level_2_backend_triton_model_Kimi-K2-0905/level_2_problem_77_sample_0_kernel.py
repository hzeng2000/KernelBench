import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_scale_bn_gi_kernel(
    x_ptr, w_ptr, b_ptr, running_mean_ptr, running_var_ptr, eps,
    out_ptr, 
    batch_size, out_channels, out_d, out_h, out_w,
    in_channels, kernel_size,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    scale_factor,
    BLOCK_C: tl.constexpr, BLOCK_DHW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    c_start = pid_c * BLOCK_C
    dhw_start = pid_dhw * BLOCK_DHW

    c_offs = c_start + tl.arange(0, BLOCK_C)
    dhw_offs = dhw_start + tl.arange(0, BLOCK_DHW)

    mask_c = c_offs < out_channels
    mask_dhw = dhw_offs < (out_d * out_h * out_w)

    # Compute output spatial indices
    d_idx = dhw_offs // (out_h * out_w)
    hw_idx = dhw_offs % (out_h * out_w)
    h_idx = hw_idx // out_w
    w_idx = hw_idx % out_w

    # Compute input spatial indices for convolution
    in_d_start = d_idx - pad_d
    in_h_start = h_idx - pad_h
    in_w_start = w_idx - pad_w

    acc = tl.zeros([BLOCK_C, BLOCK_DHW], dtype=tl.float32)

    # Convolution
    for ic in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    in_d = in_d_start + kd
                    in_h = in_h_start + kh
                    in_w = in_w_start + kw

                    # Bounds check for input
                    valid = (in_d >= 0) & (in_d < out_d) & (in_h >= 0) & (in_h < out_h) & (in_w >= 0) & (in_w < out_w)
                    if valid:
                        # Load weight
                        w_offs = ((c_offs * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw
                        w_val = tl.load(w_ptr + w_offs, mask=mask_c, other=0.0)

                        # Load input (assume stride=1, dilation=1)
                        x_offs = ((pid_b * in_channels + ic) * out_d + in_d) * out_h * out_w + in_h * out_w + in_w
                        x_val = tl.load(x_ptr + x_offs, mask=mask_dhw, other=0.0)

                        acc += w_val[:, None] * x_val[None, :]

    # Add bias
    b_offs = c_offs
    b_val = tl.load(b_ptr + b_offs, mask=mask_c, other=0.0)
    acc += b_val[:, None]

    # Scale
    acc *= scale_factor

    # Batch norm (inference mode)
    mean = tl.load(running_mean_ptr + c_offs, mask=mask_c, other=0.0)
    var = tl.load(running_var_ptr + c_offs, mask=mask_c, other=0.0)
    inv_std = tl.rsqrt(var + eps)
    acc = (acc - mean[:, None]) * inv_std[:, None]

    # Store output
    out_offs = ((pid_b * out_channels + c_offs[:, None]) * out_d + d_idx[None, :]) * out_h * out_w + h_idx[None, :] * out_w + w_idx[None, :]
    tl.store(out_ptr + out_offs, acc, mask=mask_c[:, None] & mask_dhw[None, :])


@triton.jit
def global_avg_pool3d_kernel(
    x_ptr, out_ptr,
    batch_size, channels, d, h, w,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offs < channels

    sum_val = tl.zeros([BLOCK_C], dtype=tl.float32)
    for d_idx in range(d):
        for h_idx in range(h):
            for w_idx in range(w):
                offs = ((pid_b * channels + c_offs) * d + d_idx) * h * w + h_idx * w + w_idx
                val = tl.load(x_ptr + offs, mask=mask_c, other=0.0)
                sum_val += val

    avg = sum_val / (d * h * w)
    out_offs = pid_b * channels + c_offs
    tl.store(out_ptr + out_offs, avg, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Precompute weight and bias for transposed convolution as regular convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Register running stats for batch norm
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))

    def forward(self, x):
        batch_size, in_channels, depth, height, width = x.shape
        out_channels = self.weight.shape[0]
        kernel_size = self.weight.shape[2]
        out_d = depth + kernel_size - 1
        out_h = height + kernel_size - 1
        out_w = width + kernel_size - 1

        # Allocate output
        out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

        # Grid
        BLOCK_C = 8
        BLOCK_DHW = 8
        grid = (batch_size, (out_channels + BLOCK_C - 1) // BLOCK_C, (out_d * out_h * out_w + BLOCK_DHW - 1) // BLOCK_DHW)

        fused_transpose_scale_bn_gi_kernel[grid](
            x, self.weight, self.bias, self.running_mean, self.running_var, self.batch_norm.eps,
            out,
            batch_size, out_channels, out_d, out_h, out_w,
            in_channels, kernel_size,
            1, 1, 1,  # strides
            0, 0, 0,  # padding
            self.scale_factor,
            BLOCK_C=BLOCK_C, BLOCK_DHW=BLOCK_DHW
        )

        # Global average pooling
        pooled = torch.empty(batch_size, out_channels, 1, 1, 1, device=x.device, dtype=x.dtype)
        grid_pool = (batch_size, (out_channels + BLOCK_C - 1) // BLOCK_C)
        global_avg_pool3d_kernel[grid_pool](
            out, pooled,
            batch_size, out_channels, out_d, out_h, out_w,
            BLOCK_C=BLOCK_C
        )

        return pooled.squeeze(-1).squeeze(-1).squeeze(-1)