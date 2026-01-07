import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_add_hardswish_mul_kernel(
    x_ptr,  # Pointer to conv output
    add_ptr,  # Pointer to add_input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    add_val = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    # Compute z = x + add_val
    z = x + add_val
    # Compute HardSwish: z * relu6(z + 3) / 6
    relu6 = tl.maximum(0.0, tl.minimum(6.0, z + 3.0))
    hardswish = z * relu6 / 6.0
    # Compute output = z * hardswish
    out = z * hardswish
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_add_hardswish_mul(x: torch.Tensor, add_input: torch.Tensor):
    """
    Fused kernel for addition followed by element-wise multiplication with HardSwish.
    """
    assert x.is_cuda and add_input.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    add_input = add_input.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    fused_add_hardswish_mul_kernel[grid](x, add_input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W) after HardSwish activation.
        """
        x = self.conv_transpose(x)
        # Fused addition and HardSwish multiplication using Triton kernel
        x = triton_fused_add_hardswish_mul(x, add_input)
        return x