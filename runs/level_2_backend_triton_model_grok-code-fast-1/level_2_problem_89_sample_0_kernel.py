import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_softmax_subtract_swish_max_kernel(
    input_ptr,  # Pointer to input tensor (B, C, D, H, W)
    subtract_ptr,  # Pointer to subtract parameter (C,)
    output_ptr,  # Pointer to output tensor (B, D, H, W)
    B, C, D, H, W,
):
    # Each program handles one spatial position (b, d, h, w)
    pid = tl.program_id(0)
    total_spatial = B * D * H * W
    b = pid // (D * H * W)
    rem = pid % (D * H * W)
    d = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    # Offsets for loading the C channels for this (b, d, h, w)
    offsets = b * (C * D * H * W) + tl.arange(0, C) * (D * H * W) + d * (H * W) + h * W + w
    x = tl.load(input_ptr + offsets)  # Load C values

    # Load subtract values
    sub_offsets = tl.arange(0, C)
    sub = tl.load(subtract_ptr + sub_offsets)

    # Online softmax: compute max, exp, sum, normalize
    max_val = tl.max(x)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x)
    softmax = exp_x / sum_exp

    # Subtract
    y = softmax - sub

    # Swish: sigmoid(y) * y
    sig = tl.sigmoid(y)
    swish = sig * y

    # Max over swish
    result = tl.max(swish)

    # Store result
    out_offset = b * (D * H * W) + d * (H * W) + h * W + w
    tl.store(output_ptr + out_offset, result)


def triton_fused_softmax_subtract_swish_max(x: torch.Tensor, subtract: torch.Tensor):
    """
    Fused kernel for softmax (dim=1), subtract, swish, and max (dim=1).
    """
    assert x.is_cuda and subtract.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    subtract = subtract.contiguous()

    B, C, D, H, W = x.shape
    out = torch.empty(B, D, H, W, dtype=x.dtype, device=x.device)

    grid = (B * D * H * W,)
    fused_softmax_subtract_swish_max_kernel[grid](
        x, subtract, out, B, C, D, H, W
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused Triton kernel for softmax, subtract, swish, and max.
    ConvTranspose3d and MaxPool3d remain as PyTorch ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))  # Assuming subtraction is element-wise across channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        # Fused operation: softmax(dim=1), subtract, swish, max(dim=1)
        x = triton_fused_softmax_subtract_swish_max(x, self.subtract)
        return x