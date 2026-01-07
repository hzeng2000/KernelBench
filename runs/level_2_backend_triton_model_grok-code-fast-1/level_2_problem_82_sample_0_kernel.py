import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,  # Pointer to input tensor (after conv)
    bias_ptr,  # Pointer to bias tensor (shape C)
    out_ptr,  # Pointer to output tensor
    scaling_factor,  # Scaling factor (float)
    B,  # Batch size
    C,  # Number of channels
    H,  # Height
    W,  # Width
    N_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute indices
    HW = H * W
    CHW = C * HW
    b = offsets // CHW
    local_offset = offsets % CHW
    c = local_offset // HW
    
    # Load bias for channel c
    bias = tl.load(bias_ptr + c, mask=mask)
    
    # Compute: tanh(x) * scaling_factor + bias
    out = tl.tanh(x) * scaling_factor + bias
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_post_conv(x: torch.Tensor, scaling_factor: float, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    N_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    grid = lambda meta: ((N_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_post_conv_kernel[grid](x, bias, out, scaling_factor, B, C, H, W, N_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies fused tanh+scaling+bias addition via Triton, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Fused tanh, scaling, and bias addition using Triton
        x = fused_post_conv(x, self.scaling_factor, self.bias)
        # Max-pooling
        x = self.max_pool(x)
        return x