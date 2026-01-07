import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_maxpool_clamp_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    clamp_min,
    clamp_max,
    B,
    C,
    H,
    W,
    oh,
    ow,
    ks,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    out_stride_b,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output position
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    # Load scale for this channel
    scale_val = tl.load(scale_ptr + pid_c)

    # Compute input window offsets
    ih_start = pid_oh * ks
    iw_start = pid_ow * ks
    ih_offsets = ih_start + tl.arange(0, ks)
    iw_offsets = iw_start + tl.arange(0, ks)
    mask_ih = ih_offsets < H
    mask_iw = iw_offsets < W

    # Compute offsets for loading x
    offsets = (
        pid_b * stride_b
        + pid_c * stride_c
        + ih_offsets[:, None] * stride_h
        + iw_offsets[None, :] * stride_w
    )
    x_vals = tl.load(x_ptr + offsets, mask=mask_ih[:, None] & mask_iw[None, :], other=-float("inf"))
    x_scaled = x_vals * scale_val
    max_val = tl.max(x_scaled)
    clamped = tl.clamp(max_val, clamp_min, clamp_max)

    # Store to output
    out_offset = (
        pid_b * out_stride_b
        + pid_c * out_stride_c
        + pid_oh * out_stride_h
        + pid_ow * out_stride_w
    )
    tl.store(out_ptr + out_offset, clamped)


def triton_scale_maxpool_clamp(x: torch.Tensor, scale: torch.Tensor, clamp_min: float, clamp_max: float, ks: int):
    assert x.is_cuda and scale.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    scale = scale.contiguous()

    B, C, H, W = x.shape
    oh = H // ks
    ow = W // ks
    out = torch.empty(B, C, oh, ow, device=x.device, dtype=x.dtype)

    stride_b = C * H * W
    stride_c = H * W
    stride_h = W
    stride_w = 1
    out_stride_b = C * oh * ow
    out_stride_c = oh * ow
    out_stride_h = ow
    out_stride_w = 1

    grid = (B, C, oh, ow)
    scale_maxpool_clamp_kernel[grid](
        x,
        scale,
        out,
        clamp_min,
        clamp_max,
        B,
        C,
        H,
        W,
        oh,
        ow,
        ks,
        stride_b,
        stride_c,
        stride_h,
        stride_w,
        out_stride_b,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        BLOCK_SIZE=1,  # Not used, but required
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = triton_scale_maxpool_clamp(x, self.scale, self.clamp_min, self.clamp_max, self.maxpool_kernel_size)
        return x