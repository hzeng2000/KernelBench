import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bn_scale_kernel(
    x_ptr,  # Pointer to input tensor (output of conv)
    running_mean_ptr,  # Pointer to running_mean
    running_var_ptr,  # Pointer to running_var
    weight_ptr,  # Pointer to weight
    bias_ptr,  # Pointer to bias
    out_ptr,  # Pointer to output
    scaling_factor,  # Scaling factor
    eps,  # Epsilon for BatchNorm
    channels,  # Number of channels
    height,  # Height of feature map
    width,  # Width of feature map
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total_elements = tl.numel(x_ptr)  # Total elements in the tensor
    mask = offsets < total_elements

    # Compute channel index c
    chw_per_channel = height * width
    c_per_batch = channels * chw_per_channel
    b = offsets // c_per_batch
    rem = offsets % c_per_batch
    c = rem // chw_per_channel

    # Load values
    x = tl.load(x_ptr + offsets, mask=mask)
    running_mean = tl.load(running_mean_ptr + c, mask=mask)
    running_var = tl.load(running_var_ptr + c, mask=mask)
    weight = tl.load(weight_ptr + c, mask=mask)
    bias = tl.load(bias_ptr + c, mask=mask)

    # Compute BatchNorm + scaling
    out = (x - running_mean) / tl.sqrt(running_var + eps) * weight + bias
    out = out * scaling_factor

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_bn_scale(x: torch.Tensor, bn: nn.BatchNorm2d, scaling_factor: float):
    """
    Fused BatchNorm and scaling using Triton kernel.
    Assumes the BatchNorm is in eval mode and uses running stats.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    batch_size, channels, height, width = x.shape
    BLOCK_SIZE = 128  # Tunable

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    fused_bn_scale_kernel[grid](
        x,
        bn.running_mean,
        bn.running_var,
        bn.weight,
        bn.bias,
        out,
        scaling_factor,
        bn.eps,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused BatchNorm and scaling using Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        # Fused BatchNorm and scaling
        x = triton_fused_bn_scale(x, self.bn, self.scaling_factor)
        return x