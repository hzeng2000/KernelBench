import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def post_conv_kernel(
    x_ptr,  # Pointer to input tensor (after conv)
    out_ptr,  # Pointer to output tensor
    add_value,  # Scalar to add
    scale,  # Scalar to multiply
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Mish: x * tanh(softplus(x))
    softplus = tl.log(1.0 + tl.exp(x))  # softplus(x)
    tanh_softplus = tl.tanh(softplus)   # tanh(softplus(x))
    mish = x * tanh_softplus            # mish(x)
    # Add scalar
    mish_add = mish + add_value
    # Hardtanh: clamp to [-1, 1]
    hardtanh = tl.clamp(mish_add, -1.0, 1.0)
    # Scale
    out = hardtanh * scale
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_post_conv(x: torch.Tensor, add_value: float, scale: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    post_conv_kernel[grid](x, out, add_value, scale, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value, 
    applies Hardtanh activation, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_post_conv(x, self.add_value, self.scale)
        return x