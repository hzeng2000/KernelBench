import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_mul_norm_clamp_mul_max_kernel(
    x_ptr, weight_ptr, bias_ptr, multiplier_ptr, out_ptr,
    batch_size, out_channels, out_depth, out_height, out_width,
    in_channels, kernel_size, depth, height, width,
    clamp_min, clamp_max,
    stride_d: tl.constexpr, stride_h: tl.constexpr, stride_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_elements = batch_size * out_depth * out_height * out_width
    
    for i in range(pid * BLOCK_SIZE, min((pid + 1) * BLOCK_SIZE, n_elements)):
        if i < n_elements:
            # Compute output indices
            tmp = i
            w = tmp % out_width
            tmp = tmp // out_width
            h = tmp % out_height
            tmp = tmp // out_height
            d = tmp % out_depth
            tmp = tmp // out_depth
            b = tmp % batch_size
            
            # Compute mean and var for instance norm
            sum_val = 0.0
            sum_sq = 0.0
            for c in range(out_channels):
                # Compute convolution for this output position
                conv_val = 0.0
                for ic in range(in_channels):
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                in_d = d * stride_d + kd - kernel_size // 2
                                in_h = h * stride_h + kh - kernel_size // 2
                                in_w = w * stride_w + kw - kernel_size // 2
                                if 0 <= in_d < depth and 0 <= in_h < height and 0 <= in_w < width:
                                    in_idx = ((b * in_channels + ic) * depth + in_d) * height * width + in_h * width + in_w
                                    weight_idx = ((c * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw
                                    conv_val += tl.load(x_ptr + in_idx) * tl.load(weight_ptr + weight_idx)
                
                # Add bias
                conv_val += tl.load(bias_ptr + c)
                
                # Multiply by multiplier
                conv_val *= tl.load(multiplier_ptr + c)
                
                sum_val += conv_val
                sum_sq += conv_val * conv_val
            
            # Compute instance norm
            mean = sum_val / out_channels
            var = sum_sq / out_channels - mean * mean
            std = tl.sqrt(var + 1e-5)
            
            # Apply operations and store max
            max_val = -float('inf')
            for c in range(out_channels):
                # Compute convolution
                conv_val = 0.0
                for ic in range(in_channels):
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                in_d = d * stride_d + kd - kernel_size // 2
                                in_h = h * stride_h + kh - kernel_size // 2
                                in_w = w * stride_w + kw - kernel_size // 2
                                if 0 <= in_d < depth and 0 <= in_h < height and 0 <= in_w < width:
                                    in_idx = ((b * in_channels + ic) * depth + in_d) * height * width + in_h * width + in_w
                                    weight_idx = ((c * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size + kh * kernel_size + kw
                                    conv_val += tl.load(x_ptr + in_idx) * tl.load(weight_ptr + weight_idx)
                
                # Add bias
                conv_val += tl.load(bias_ptr + c)
                
                # Multiply by multiplier
                conv_val *= tl.load(multiplier_ptr + c)
                
                # Instance norm
                norm_val = (conv_val - mean) / std
                
                # Clamp
                clamped = tl.clamp(norm_val, clamp_min, clamp_max)
                
                # Multiply by multiplier again
                final_val = clamped * tl.load(multiplier_ptr + c)
                
                # Track max
                max_val = tl.maximum(max_val, final_val)
            
            # Store result
            out_idx = ((b * out_depth + d) * out_height + h) * out_width + w
            tl.store(out_ptr + out_idx, max_val)


def triton_fused_conv_mul_norm_clamp_mul_max(x, weight, bias, multiplier, clamp_min, clamp_max):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    
    # Compute output dimensions (assuming stride=1, padding=kernel_size//2)
    out_depth = depth
    out_height = height
    out_width = width
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Flatten tensors
    x_flat = x.contiguous().view(-1)
    weight_flat = weight.contiguous().view(-1)
    bias_flat = bias.contiguous().view(-1)
    multiplier_flat = multiplier.contiguous().view(-1)
    out_flat = out.view(-1)
    
    n_elements = batch_size * out_depth * out_height * out_width
    BLOCK_SIZE = 128
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_conv_mul_norm_clamp_mul_max_kernel[grid](
        x_flat, weight_flat, bias_flat, multiplier_flat, out_flat,
        batch_size, out_channels, out_depth, out_height, out_width,
        in_channels, kernel_size, depth, height, width,
        clamp_min, clamp_max,
        stride_d=1, stride_h=1, stride_w=1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        return triton_fused_conv_mul_norm_clamp_mul_max(
            x, self.conv.weight, self.conv.bias, self.multiplier,
            self.clamp_min, self.clamp_max
        )