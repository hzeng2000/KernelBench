import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_layernorm_gelu_scale_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    scaling_factor,
    eps,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    # Compute var
    x_minus_mean = x - mean
    var = tl.sum(x_minus_mean * x_minus_mean, axis=0) / N
    # rstd
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize
    x_norm = x_minus_mean * rstd
    # Affine
    w = tl.load(weight_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + offsets, mask=mask)
    x_affine = x_norm * w + b
    # GELU
    x_gelu = 0.5 * x_affine * (1 + tl.erf(x_affine / tl.sqrt(2.0)))
    # Scale
    x_final = x_gelu * scaling_factor
    tl.store(out_ptr + row_start + offsets, x_final, mask=mask)


def triton_fused_layernorm_gelu_scale(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scaling_factor: float, eps: float):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    original_shape = x.shape
    x = x.contiguous().view(-1, original_shape[-1])
    weight = weight.contiguous().view(-1)
    bias = bias.contiguous().view(-1)
    out = torch.empty_like(x)
    num_rows, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (num_rows,)
    fused_layernorm_gelu_scale_kernel[grid](
        x, weight, bias, out, scaling_factor, eps, N, BLOCK_SIZE=BLOCK_SIZE
    )
    out = out.view(original_shape)
    return out


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
        x = self.conv_transpose(x)
        x = triton_fused_layernorm_gelu_scale(x, self.layer_norm.weight, self.layer_norm.bias, self.scaling_factor, self.layer_norm.eps)
        return x