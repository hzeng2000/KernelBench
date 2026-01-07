import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_activation_kernel(
    x_ptr,  # Pointer to input (conv output)
    bias_ptr,  # Pointer to bias
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    d, h, w, c,  # Dimensions: depth, height, width, channels
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index for bias
    stride_cd = d * h * w
    channel = (offsets // stride_cd) % c
    bias_val = tl.load(bias_ptr + channel)  # Bias is small, no mask needed
    
    # Apply ReLU
    out = tl.maximum(x, 0.0)
    
    # Apply LeakyReLU on the ReLU result
    out = tl.where(out > 0.0, out, 0.01 * out)
    
    # Apply GELU
    sqrt2 = tl.sqrt(2.0)
    erf_arg = out / sqrt2
    out = 0.5 * out * (1.0 + tl.erf(erf_arg))
    
    # Apply Sigmoid
    out = 1.0 / (1.0 + tl.exp(-out))
    
    # Add bias
    out = out + bias_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_activations(x: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper for the fused activation kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    batch, c, d, h, w = x.shape
    BLOCK_SIZE = 1024  # Tunable block size
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_activation_kernel[grid](
        x, bias, out, n_elements, d, h, w, c, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, then applies fused ReLU, LeakyReLU, GELU, Sigmoid, and bias addition using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = triton_fused_activations(x, self.bias)
        return x