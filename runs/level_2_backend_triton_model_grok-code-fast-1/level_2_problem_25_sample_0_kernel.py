import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_tanh_kernel(
    in_ptr,
    out_ptr,
    B, C, H, W,
):
    pid = tl.program_id(0)
    # Compute b, h, w
    total_hw = H * W
    b = pid // total_hw
    hw = pid % total_hw
    h = hw // W
    w = hw % W
    # Load the C values and find min
    min_val = float('inf')
    for c in range(C):
        offset = b * (C * H * W) + c * (H * W) + h * W + w
        val = tl.load(in_ptr + offset)
        min_val = tl.minimum(min_val, val)
    # Apply tanh twice
    out_val = tl.tanh(tl.tanh(min_val))
    # Store
    tl.store(out_ptr + pid, out_val)


def fused_min_tanh(x: torch.Tensor):
    assert x.is_cuda and x.is_contiguous()
    B, C, H, W = x.shape
    out = torch.empty(B, 1, H, W, dtype=x.dtype, device=x.device)
    grid = (B * H * W,)
    min_tanh_kernel[grid](x, out, B, C, H, W)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = fused_min_tanh(x)
        return x