import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_elements = batch_size * out_channels * out_d * out_h * out_w
    if pid >= out_elements:
        return
    
    # Compute output coordinates
    out_idx = pid
    out_w_idx = out_idx % out_w
    out_idx = out_idx // out_w
    out_h_idx = out_idx % out_h
    out_idx = out_idx // out_h
    out_d_idx = out_idx % out_d
    out_idx = out_idx // out_d
    out_c_idx = out_idx % out_channels
    out_idx = out_idx // out_channels
    b_idx = out_idx
    
    # Compute input start position
    in_d_start = out_d_idx * stride_d - padding_d
    in_h_start = out_h_idx * stride_h - padding_h
    in_w_start = out_w_idx * stride_w - padding_w
    
    acc = 0.0
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_d_idx = in_d_start + kd
                in_h_idx = in_h_start + kh
                in_w_idx = in_w_start + kw
                
                if in_d_idx >= 0 and in_d_idx < in_d and in_h_idx >= 0 and in_h_idx < in_h and in_w_idx >= 0 and in_w_idx < in_w:
                    for ic in range(in_channels):
                        in_offset = b_idx * in_channels * in_d * in_h * in_w + ic * in_d * in_h * in_w + in_d_idx * in_h * in_w + in_h_idx * in_w + in_w_idx
                        w_offset = out_c_idx * in_channels * kernel_d * kernel_h * kernel_w + ic * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw
                        in_val = tl.load(input_ptr + in_offset)
                        w_val = tl.load(weight_ptr + w_offset)
                        acc += in_val * w_val
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c_idx)
        acc += bias_val
    
    out_offset = b_idx * out_channels * out_d * out_h * out_w + out_c_idx * out_d * out_h * out_w + out_d_idx * out_h * out_w + out_h_idx * out_w + out_w_idx
    tl.store(output_ptr + out_offset, acc)


@triton.jit
def layer_norm_gelu_scale_kernel(
    x_ptr, out_ptr, gamma_ptr, beta_ptr, scaling_factor,
    batch_size, channels, d, h, w,
    eps, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * d * h * w
    if pid >= total_elements:
        return
    
    # Compute spatial coordinates
    spatial_idx = pid
    w_idx = spatial_idx % w
    spatial_idx = spatial_idx // w
    h_idx = spatial_idx % h
    spatial_idx = spatial_idx // h
    d_idx = spatial_idx % d
    b_idx = spatial_idx // d
    
    # Compute mean and variance
    mean = 0.0
    for c in range(channels):
        offset = b_idx * channels * d * h * w + c * d * h * w + d_idx * h * w + h_idx * w + w_idx
        val = tl.load(x_ptr + offset)
        mean += val
    mean /= channels
    
    var = 0.0
    for c in range(channels):
        offset = b_idx * channels * d * h * w + c * d * h * w + d_idx * h * w + h_idx * w + w_idx
        val = tl.load(x_ptr + offset)
        var += (val - mean) * (val - mean)
    var /= channels
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Apply layer norm, GELU, and scaling
    for c in range(channels):
        offset = b_idx * channels * d * h * w + c * d * h * w + d_idx * h * w + h_idx * w + w_idx
        val = tl.load(x_ptr + offset)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        norm_val = (val - mean) * rstd
        norm_val = norm_val * gamma + beta
        
        # GELU approximation
        gelu_val = 0.5 * norm_val * (1.0 + tl.tanh(0.7978845608 * (norm_val + 0.044715 * norm_val * norm_val * norm_val)))
        
        # Scale
        scaled_val = gelu_val * scaling_factor
        tl.store(out_ptr + offset, scaled_val)


def triton_conv_transpose3d(x, weight, bias, stride, padding):
    batch_size, in_channels, in_d, in_h, in_w = x.shape
    out_channels, _, kernel_d, kernel_h, kernel_w = weight.shape
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    
    out_d = (in_d - 1) * stride_d - 2 * padding_d + kernel_d
    out_h = (in_h - 1) * stride_h - 2 * padding_h + kernel_h
    out_w = (in_w - 1) * stride_w - 2 * padding_w + kernel_w
    
    output = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)
    
    out_elements = batch_size * out_channels * out_d * out_h * out_w
    BLOCK_SIZE = 128
    grid = lambda meta: ((out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    conv_transpose_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def triton_layer_norm_gelu_scale(x, gamma, beta, scaling_factor, eps):
    batch_size, channels, d, h, w = x.shape
    output = torch.empty_like(x)
    
    total_elements = batch_size * d * h * w
    BLOCK_SIZE = 128
    grid = lambda meta: ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    layer_norm_gelu_scale_kernel[grid](
        x, output, gamma, beta, scaling_factor,
        batch_size, channels, d, h, w,
        eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = triton_conv_transpose3d(x, self.conv_transpose.weight, self.conv_transpose.bias, self.conv_transpose.stride, self.conv_transpose.padding)
        x = triton_layer_norm_gelu_scale(x, self.layer_norm.weight, self.layer_norm.bias, self.scaling_factor, self.layer_norm.eps)
        return x