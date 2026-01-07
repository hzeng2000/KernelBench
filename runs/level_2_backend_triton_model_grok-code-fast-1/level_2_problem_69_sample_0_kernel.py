import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardswish_relu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute HardSwish: x * relu6(x + 3) / 6
    relu6_arg = x + 3.0
    relu6 = tl.maximum(tl.minimum(relu6_arg, 6.0), 0.0)
    hardswish = x * relu6 / 6.0
    # Apply ReLU (though HardSwish is already >= 0, to match the original)
    out = tl.maximum(hardswish, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardswish_relu(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardswish_relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, applies HardSwish, and then ReLU using a fused Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = triton_hardswish_relu(x)
        return x