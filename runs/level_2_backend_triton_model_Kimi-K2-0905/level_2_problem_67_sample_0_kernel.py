import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for Conv2d + GELU fused
@triton.jit
def conv_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_c, in_h, in_w,
    out_c, out_h, out_w,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute output pixel coordinates
    hw = pid_hw * BLOCK_H * BLOCK_W + tl.arange(0, BLOCK_H * BLOCK_W)
    oh = hw // out_w
    ow = hw % out_w

    # Compute input window top-left corner
    ih0 = oh * stride_h - pad_h
    iw0 = ow * stride_w - pad_w

    # Allocate accumulator
    acc = tl.zeros([BLOCK_H * BLOCK_W], dtype=tl.float32)

    # Loop over input channels
    for ic in range(in_c):
        # Compute input channel offset
        x_offset = pid_b * in_c * in_h * in_w + ic * in_h * in_w
        # Compute weight offset for this output channel
        w_offset = pid_c * in_c * k_h * k_w + ic * k_h * k_w

        # Load weights for this input channel
        w = tl.load(w_ptr + w_offset + tl.arange(0, k_h * k_w))

        # Loop over kernel window
        for kh in range(k_h):
            for kw in range(k_w):
                ih = ih0 + kh
                iw = iw0 + kw
                # Bounds check
                mask = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
                # Load input
                x_idx = x_offset + ih * in_w + iw
                x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                # Load weight
                w_val = w[kh * k_w + kw]
                # Accumulate
                acc += x_val * w_val

    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + pid_c)
        acc += b_val

    # Apply GELU
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pi = 3.141592653589793
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    cube = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + coeff * cube)
    tanh_inner = tl.tanh(inner)
    gelu_out = 0.5 * acc * (1.0 + tanh_inner)

    # Compute output offset
    out_offset = pid_b * out_c * out_h * out_w + pid_c * out_h * out_w
    out_idx = out_offset + oh * out_w + ow
    # Store
    tl.store(out_ptr + out_idx, gelu_out, mask=hw < out_h * out_w)


# Triton kernel for global average pooling (adaptive_avg_pool2d to 1x1)
@triton.jit
def global_avg_pool_kernel(
    x_ptr, out_ptr,
    batch, channels, height, width,
    BLOCK_CHANNELS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Compute channel offset
    x_offset = pid_b * channels * height * width + pid_c * height * width
    # Load all spatial elements for this channel
    hw = tl.arange(0, height * width)
    mask = hw < height * width
    x_val = tl.load(x_ptr + x_offset + hw, mask=mask, other=0.0)
    # Compute mean
    sum_val = tl.sum(x_val)
    mean_val = sum_val / (height * width)

    # Store
    out_offset = pid_b * channels
    tl.store(out_ptr + out_offset + pid_c, mean_val)


def triton_conv_gelu(x, weight, bias, stride=1, padding=0):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch, in_c, in_h, in_w = x.shape
    out_c, _, k_h, k_w = weight.shape
    stride_h = stride_w = stride
    pad_h = pad_w = padding

    # Compute output spatial dims
    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1

    out = torch.empty((batch, out_c, out_h, out_w), dtype=torch.float32, device=x.device)

    # Grid dimensions
    BLOCK_BATCH = 1
    BLOCK_OUT_C = 1
    BLOCK_H = 4
    BLOCK_W = 4

    grid = (
        batch,
        out_c,
        (out_h * out_w + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W),
    )

    conv_gelu_kernel[grid](
        x, weight, bias, out,
        batch, in_c, in_h, in_w,
        out_c, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OUT_C=BLOCK_OUT_C,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out


def triton_global_avg_pool(x):
    assert x.is_cuda
    x = x.contiguous()
    batch, channels, height, width = x.shape
    out = torch.empty((batch, channels), dtype=torch.float32, device=x.device)

    BLOCK_CHANNELS = 1
    grid = (batch, channels)

    global_avg_pool_kernel[grid](
        x, out,
        batch, channels, height, width,
        BLOCK_CHANNELS=BLOCK_CHANNELS,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels.
    Performs fused Conv2d + GELU and custom global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Store kernel size and padding for use in forward
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Same padding

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        weight = self.conv.weight
        bias = self.conv.bias
        # Fused conv + GELU
        x = triton_conv_gelu(x, weight, bias, stride=1, padding=self.padding)
        # Custom global average pooling
        x = triton_global_avg_pool(x)
        return x