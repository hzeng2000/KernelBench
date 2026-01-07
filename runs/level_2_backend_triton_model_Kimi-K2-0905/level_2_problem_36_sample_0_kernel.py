import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_min_sum_gelu_add_kernel(
    x_ptr,      # pointer to input tensor after conv_transpose
    bias_ptr,   # pointer to bias
    out_ptr,    # pointer to output
    B, C, H, W, # dimensions
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute flat indices for output tensor (B, 1, 1, W)
    total_out = B * 1 * 1 * W
    mask = offsets < total_out

    # Map flat index back to output tensor indices
    out_w = offsets % W
    out_tmp = offsets // W
    out_b = out_tmp  # since C and H are 1

    # Map to input tensor indices
    # Input tensor has shape (B, C, H, W)
    # We need to compute min over C and sum over H

    min_val = float('inf')
    sum_val = 0.0

    # Iterate over C and H to compute min and sum
    for c in range(C):
        for h in range(H):
            in_idx = out_b * stride_b + c * stride_c + h * stride_h + out_w * stride_w
            val = tl.load(x_ptr + in_idx)
            if val < min_val:
                min_val = val
            sum_val += min_val

    # Apply GELU
    gelu_out = 0.5 * sum_val * (1.0 + tl.math.tanh(0.7978845608 * (sum_val + 0.044715 * sum_val * sum_val * sum_val)))

    # Add bias
    bias_val = tl.load(bias_ptr)
    out = gelu_out + bias_val

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_min_sum_gelu_add(x, bias):
    B, C, H, W = x.shape
    out = torch.empty((B, 1, 1, W), dtype=x.dtype, device=x.device)

    n_elements = out.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    stride_b = C * H * W
    stride_c = H * W
    stride_h = W
    stride_w = 1

    fused_min_sum_gelu_add_kernel[grid](
        x, bias, out,
        B, C, H, W,
        stride_b, stride_c, stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_fused_min_sum_gelu_add(x, self.bias)
        return x