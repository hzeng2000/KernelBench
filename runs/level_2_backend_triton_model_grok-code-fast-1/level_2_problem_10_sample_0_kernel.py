import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_hardtanh_mean_tanh_kernel(
    x_ptr,  # Pointer to input tensor (after maxpool)
    out_ptr,  # Pointer to output tensor
    B, C, H, W,  # Dimensions
    hardtanh_min, hardtanh_max,
):
    b_c = tl.program_id(0)
    b = b_c // C
    c = b_c % C
    
    sum_val = 0.0
    for h in range(H):
        offsets = b * (C * H * W) + c * (H * W) + h * W + tl.arange(0, W)
        vals = tl.load(x_ptr + offsets)
        vals = tl.clamp(vals, hardtanh_min, hardtanh_max)
        sum_val += tl.sum(vals)
    
    mean_val = sum_val / (H * W)
    tanh_val = tl.tanh(mean_val)
    
    out_idx = b * (C * 1 * 1) + c * (1 * 1) + 0 * 1 + 0
    tl.store(out_ptr + out_idx, tanh_val)


def fused_hardtanh_mean_tanh(x: torch.Tensor, hardtanh_min: float, hardtanh_max: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    B, C, H, W = x.shape
    out = torch.empty(B, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = (B * C,)
    fused_hardtanh_mean_tanh_kernel[grid](x, out, B, C, H, W, hardtanh_min, hardtanh_max)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, and a fused hardtanh + mean + tanh operation using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = fused_hardtanh_mean_tanh(x, self.hardtanh_min, self.hardtanh_max)
        return x