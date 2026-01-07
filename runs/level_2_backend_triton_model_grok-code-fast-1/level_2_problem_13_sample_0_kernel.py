import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def post_conv_kernel(
    x_ptr,  # Pointer to conv output: (B, C, D, H, W)
    bias_ptr,  # Pointer to bias: (1, C, 1, 1, 1), but treated as (C,) in kernel
    out_ptr,  # Pointer to output: (B, C, 1, H, W)
    scaling_factor,  # Scaling factor
    B, C, D, H, W,  # Dimensions
    stride_b, stride_c, stride_d, stride_h, stride_w,  # Strides for x
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,  # Strides for out
    BLOCK_SIZE_C: tl.constexpr,
):
    # Grid: (B, H, W)
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    offsets_c = tl.arange(0, BLOCK_SIZE_C)
    
    # For each c, compute mean over d
    vals = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    for i in range(BLOCK_SIZE_C):
        c = offsets_c[i]
        sum_val = 0.0
        for d in range(D):
            offset = (b * stride_b + c * stride_c + d * stride_d + 
                      h * stride_h + w * stride_w)
            sum_val += tl.load(x_ptr + offset)
        mean_val = sum_val / D
        bias_val = tl.load(bias_ptr + c)
        vals = tl.where(offsets_c == c, mean_val + bias_val, vals)
    
    # Compute softmax
    max_val = tl.reduce(vals, tl.maximum)
    exp_vals = tl.exp(vals - max_val)
    sum_exp = tl.reduce(exp_vals, tl.add)
    softmax_vals = exp_vals / sum_exp
    
    # Apply tanh and scaling
    out_vals = tl.tanh(softmax_vals) * scaling_factor
    
    # Store
    for i in range(BLOCK_SIZE_C):
        c = offsets_c[i]
        out_offset = (b * out_stride_b + c * out_stride_c + 0 * out_stride_d + 
                      h * out_stride_h + w * out_stride_w)
        tl.store(out_ptr + out_offset, out_vals[i])


def triton_post_conv(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    
    B, C, D, H, W = x.shape
    out = torch.empty(B, C, 1, H, W, dtype=x.dtype, device=x.device)
    
    stride_b, stride_c, stride_d, stride_h, stride_w = x.stride()
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w = out.stride()
    
    BLOCK_SIZE_C = C  # 64
    
    grid = (B, H, W)
    
    post_conv_kernel[grid](
        x, bias, out, scaling_factor,
        B, C, D, H, W,
        stride_b, stride_c, stride_d, stride_h, stride_w,
        out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a series of operations:
    1. Transposed 3D convolution
    2. Fused mean pooling, bias add, softmax, tanh, scaling via Triton kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))  # Broadcastable bias over channels
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)                            # (B, C, D, H, W)
        x = triton_post_conv(x, self.bias, self.scaling_factor)  # Fused operations
        return x