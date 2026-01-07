import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_conv_bias_clamp_scale_clamp_divide_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    kernel_size, stride, padding, output_padding,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    numel = batch_size * out_channels * out_h * out_w
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute indices
    idx = offsets
    w_out = idx % out_w
    idx = idx // out_w
    h_out = idx % out_h
    idx = idx // out_h
    c_out = idx % out_channels
    idx = idx // out_channels
    b = idx
    
    # Compute input region
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding
    h_in_end = h_in_start + kernel_size
    w_in_end = w_in_start + kernel_size
    
    acc = 0.0
    for c_in in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                h_in = h_in_start + kh
                w_in = w_in_start + kw
                if h_in >= 0 and h_in < in_h and w_in >= 0 and w_in < in_w:
                    x_idx = ((b * in_channels + c_in) * in_h + h_in) * in_w + w_in
                    w_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(weight_ptr + w_idx)
                    acc += x_val * w_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + c_out)
    acc += bias_val
    
    # Clamp, scale, clamp, divide
    acc = tl.clamp(acc, 0.0, 1.0)
    acc = acc * scaling_factor
    acc = tl.clamp(acc, 0.0, 1.0)
    acc = acc / scaling_factor
    
    # Store output
    out_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out
    tl.store(out_ptr + out_idx, acc, mask=mask)


def triton_transpose_conv_bias_clamp_scale_clamp_divide(
    x, weight, bias, scaling_factor,
    stride, padding, output_padding
):
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_h = (in_h - 1) * stride - 2 * padding + kernel_size + output_padding
    out_w = (in_w - 1) * stride - 2 * padding + kernel_size + output_padding
    
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    numel = batch_size * out_channels * out_h * out_w
    BLOCK_SIZE = 128
    grid = lambda meta: ((numel + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_transpose_conv_bias_clamp_scale_clamp_divide_kernel[grid](
        x.contiguous(), weight.contiguous(), bias.contiguous(), out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_size, stride, padding, output_padding,
        scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return triton_transpose_conv_bias_clamp_scale_clamp_divide(
            x, self.conv_transpose.weight, self.bias, self.scaling_factor,
            self.conv_transpose.stride[0], self.conv_transpose.padding[0], self.conv_transpose.output_padding[0]
        )