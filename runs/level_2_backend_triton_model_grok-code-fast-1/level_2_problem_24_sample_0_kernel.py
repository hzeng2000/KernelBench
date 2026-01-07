import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_kernel(
    x_ptr,
    out_ptr,
    B, C, D, H, W,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (C * H * W)
    rem = pid % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W
    stride_b = C * D * H * W
    stride_c = D * H * W
    stride_d = H * W
    stride_h = W
    stride_w = 1
    d_offsets = tl.arange(0, BLOCK_D)
    mask = d_offsets < D
    indices = b * stride_b + c * stride_c + d_offsets * stride_d + h * stride_h + w * stride_w
    x_vals = tl.load(x_ptr + indices, mask=mask, other=float('inf'))
    min_val = tl.reduce(x_vals, tl.min, dims=[0])
    out_index = b * (C * H * W) + c * (H * W) + h * W + w
    tl.store(out_ptr + out_index, min_val)


def triton_min(x, dim):
    assert dim == 2, "Only dim=2 supported"
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    B, C, D, H, W = x.shape
    out = torch.empty(B, C, H, W, dtype=x.dtype, device=x.device)
    grid = (B * C * H * W,)
    min_kernel[grid](x, out, B, C, D, H, W, BLOCK_D=D)
    return out


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    B, C, H, W,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (H * W)
    rem = pid % (H * W)
    h = rem // W
    w = rem % W
    stride_b = C * H * W
    stride_c = H * W
    stride_h = W
    stride_w = 1
    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < C
    indices = b * stride_b + c_offsets * stride_c + h * stride_h + w * stride_w
    x_vals = tl.load(x_ptr + indices, mask=mask, other=0.0)
    max_val = tl.reduce(x_vals, tl.max, dims=[0])
    exp_vals = tl.exp(x_vals - max_val)
    sum_exp = tl.reduce(exp_vals, tl.sum, dims=[0])
    softmax_vals = exp_vals / sum_exp
    tl.store(out_ptr + indices, softmax_vals, mask=mask)


def triton_softmax(x, dim):
    assert dim == 1, "Only dim=1 supported"
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    grid = (B * H * W,)
    softmax_kernel[grid](x, out, B, C, H, W, BLOCK_C=C)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        x = triton_min(x, dim=self.dim)  # Apply minimum along the specified dimension
        x = triton_softmax(x, dim=1)  # Apply softmax along the channel dimension
        return x