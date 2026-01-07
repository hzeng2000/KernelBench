import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr,  # input, shape (batch, out_channels, H, W)
    bias_ptr,  # bias, shape (out_channels, 1, 1)
    scaling_factor,  # scalar
    out_ptr,  # output, same shape as x
    batch, out_channels, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # pid for the spatial position, flattened
    pid = tl.program_id(0)
    # compute batch_idx, h, w
    total_spatial = H * W
    batch_idx = pid // total_spatial
    spatial_idx = pid % total_spatial
    h = spatial_idx // W
    w = spatial_idx % W
    # base offset
    base_offset = batch_idx * (out_channels * H * W) + h * W + w
    offsets = base_offset + tl.arange(0, BLOCK_SIZE) * (H * W)
    # mask for channels
    mask = tl.arange(0, BLOCK_SIZE) < out_channels
    # load x
    x = tl.load(x_ptr + offsets, mask=mask)
    # load bias
    bias_offsets = tl.arange(0, BLOCK_SIZE)
    bias_mask = bias_offsets < out_channels
    bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask)
    # compute softmax: max, exp, sum, divide
    max_x = tl.max(x)
    exp_x = tl.exp(x - max_x)
    sum_exp = tl.sum(exp_x)
    softmax = exp_x / sum_exp
    # add bias
    softmax_bias = softmax + bias
    # scale
    scaled = softmax_bias * scaling_factor
    # sigmoid
    sigmoid_out = tl.sigmoid(scaled)
    # store
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)


def fused_softmax_bias_scale_sigmoid(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)
    batch, out_channels, H, W = x.shape
    BLOCK_SIZE = out_channels
    total_positions = batch * H * W
    grid = (total_positions,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias, scaling_factor, out, batch, out_channels, H, W, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies softmax, adds a bias term, scales the result, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_softmax_bias_scale_sigmoid(x, self.bias, self.scaling_factor)