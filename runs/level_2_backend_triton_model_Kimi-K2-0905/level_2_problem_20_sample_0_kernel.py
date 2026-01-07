import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_bias_residual_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Compute flat indices
    numel = B * C * D * H * W
    mask = offsets < numel
    
    # Compute 5D indices from flat offset
    b = offsets // (C * D * H * W)
    rem = offsets % (C * D * H * W)
    c = rem // (D * H * W)
    rem = rem % (D * H * W)
    d = rem // (H * W)
    rem = rem % (H * W)
    h = rem // W
    w = rem % W
    
    # Load x
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load bias (broadcasted over B, D, H, W)
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # Compute: (x + bias) + x = 2*x + bias
    out_val = 2.0 * x_val + bias_val
    # Then: out * x = (2*x + bias)*x
    out_val = out_val * x_val
    # Then: out + x = (2*x + bias)*x + x = 2*x^2 + bias*x + x
    out_val = out_val + x_val
    
    tl.store(out_ptr + offsets, out_val, mask=mask)


def fused_transpose_bias_residual_mul_add(x, bias):
    assert x.is_cuda and bias.is_cuda
    assert x.ndim == 5
    B, C, D, H, W = x.shape
    assert bias.shape == (C, 1, 1, 1)
    
    out = torch.empty_like(x)
    numel = x.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((numel + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_transpose_bias_residual_kernel[grid](
        x, bias, out,
        B, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_transpose_bias_residual_mul_add(x, self.bias)