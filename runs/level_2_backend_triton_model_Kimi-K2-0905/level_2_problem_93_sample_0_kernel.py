import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_transpose_conv_add_min_gelu_mul_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    in_height, in_width, out_height, out_width,
    kernel_size, stride, add_value, multiply_value,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate output pixel coordinates
    out_hw = out_height * out_width
    out_chw = out_channels * out_hw
    
    n = pid // out_chw
    c = (pid % out_chw) // out_hw
    hw = pid % out_hw
    h = hw // out_width
    w = hw % out_width
    
    if n >= batch_size:
        return
    
    # Compute input region
    in_h_start = h * stride
    in_w_start = w * stride
    
    # Initialize accumulator
    acc = 0.0
    
    # Perform transposed convolution
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Input coordinates
                in_h = in_h_start + kh
                in_w = in_w_start + kw
                
                # Check bounds
                if in_h < in_height and in_w < in_width:
                    # Load input
                    x_idx = n * in_channels * in_height * in_width + ic * in_height * in_width + in_h * in_width + in_w
                    x_val = tl.load(x_ptr + x_idx)
                    
                    # Load weight
                    w_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw
                    w_val = tl.load(w_ptr + w_idx)
                    
                    acc += x_val * w_val
    
    # Add bias
    if b_ptr is not None:
        b_val = tl.load(b_ptr + c)
        acc += b_val
    
    # Add value
    acc += add_value
    
    # Min with 0
    acc = tl.minimum(acc, 0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using tanh approximation for GELU
    pi = 3.14159265359
    sqrt_2_over_pi = tl.sqrt(2.0 / pi)
    x_cubed = acc * acc * acc
    tanh_arg = sqrt_2_over_pi * (acc + 0.044715 * x_cubed)
    
    # Approximate tanh using a rational function
    tanh_approx = tanh_arg / (1.0 + tl.abs(tanh_arg))
    
    gelu_out = 0.5 * acc * (1.0 + tanh_approx)
    
    # Multiply
    final_out = gelu_out * multiply_value
    
    # Store output
    out_idx = n * out_channels * out_height * out_width + c * out_height * out_width + h * out_width + w
    tl.store(out_ptr + out_idx, final_out)


def triton_transpose_conv_add_min_gelu_mul(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor,
    stride: int,
    add_value: float,
    multiply_value: float
):
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Calculate output dimensions
    out_height = (in_height - 1) * stride + kernel_size
    out_width = (in_width - 1) * stride + kernel_size
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Flatten tensors
    x_flat = x.reshape(-1)
    weight_flat = weight.reshape(-1)
    bias_flat = bias.reshape(-1) if bias is not None else None
    out_flat = out.reshape(-1)
    
    # Calculate grid
    num_elements = batch_size * out_channels * out_height * out_width
    BLOCK_SIZE = 256
    
    grid = lambda meta: (num_elements,)
    
    # Launch kernel
    fused_transpose_conv_add_min_gelu_mul_kernel[grid](
        x_flat, weight_flat, bias_flat, out_flat,
        batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, add_value, multiply_value,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        return triton_transpose_conv_add_min_gelu_mul(
            x, 
            self.conv_transpose.weight, 
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.add_value,
            self.multiply_value
        )