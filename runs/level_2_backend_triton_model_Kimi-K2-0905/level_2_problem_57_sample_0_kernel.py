import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_relu_hardswish_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate output position
    out_hw = height * width
    out_chw = out_channels * out_hw
    n = pid // out_chw
    c = (pid // out_hw) % out_channels
    hw = pid % out_hw
    h = hw // width
    w = hw % width
    
    if n >= batch_size:
        return
    
    # Compute convolution
    acc = 0.0
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = h * stride - padding + kh
                iw = w * stride - padding + kw
                if ih >= 0 and ih < height and iw >= 0 and iw < width:
                    input_idx = n * in_channels * height * width + ic * height * width + ih * width + iw
                    weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += input_val * weight_val
    
    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Apply HardSwish
    hardswish = acc * tl.clamp((acc + 3.0) / 6.0, 0.0, 1.0)
    
    # Store output
    output_idx = n * out_channels * height * width + c * height * width + h * width + w
    tl.store(output_ptr + output_idx, hardswish)


def triton_conv_relu_hardswish(input, weight, bias=None, stride=1, padding=0):
    batch_size, in_channels, height, width = input.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    output = torch.empty(batch_size, out_channels, height, width, device=input.device, dtype=input.dtype)
    
    n_elements = batch_size * out_channels * height * width
    BLOCK_SIZE = 128
    
    grid = lambda meta: (n_elements,)
    
    conv_relu_hardswish_kernel[grid](
        input, weight, bias, output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return triton_conv_relu_hardswish(x, weight, bias, stride=1, padding=0)