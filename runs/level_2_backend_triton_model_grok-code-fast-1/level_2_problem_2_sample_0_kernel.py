import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def post_process_kernel(
    x_ptr,
    bias_ptr,
    scale,
    out_ptr,
    batch,
    channels,
    height,
    width,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute indices
    elements_per_batch = channels * height * width
    elements_per_channel = height * width

    batch_idx = offsets // elements_per_batch
    rem = offsets % elements_per_batch
    c_idx = rem // elements_per_channel
    rem2 = rem % elements_per_channel
    h_idx = rem2 // width
    w_idx = rem2 % width

    # Offset for x and out
    offset = batch_idx * elements_per_batch + c_idx * elements_per_channel + h_idx * width + w_idx

    # Load x
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    # Load bias (broadcasted)
    bias_offset = c_idx  # since bias is (channels, 1, 1)
    b = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)

    # Compute
    out_val = x + b
    out_val = tl.maximum(0.0, tl.minimum(1.0, out_val))
    out_val = out_val * scale
    out_val = tl.maximum(0.0, tl.minimum(1.0, out_val))
    out_val = out_val / scale

    # Store
    tl.store(out_ptr + offset, out_val, mask=mask)


def triton_post_process(x: torch.Tensor, bias: torch.Tensor, scale: float):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()

    batch, channels, height, width = x.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    post_process_kernel[grid](
        x, bias, scale, out, batch, channels, height, width, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_post_process(x, self.bias, self.scaling_factor)
        return x