import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_sigmoid_kernel(
    x_ptr,
    out_ptr,
    B, C, D, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one (b, d, h, w)
    b = tl.program_id(0)
    d = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    
    # Offsets for channels
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    mask = c_offsets < C
    
    # Compute offsets
    x_offsets = (b * stride_b + c_offsets * stride_c + 
                 d * stride_d + h * stride_h + w * stride_w)
    
    # Load x
    x = tl.load(x_ptr + x_offsets, mask=mask, other=float('-inf'))
    
    # Compute max
    max_val = tl.max(x, axis=0)
    
    # Subtract max
    x_shift = x - max_val
    
    # Exp
    exp_x = tl.exp(x_shift)
    
    # Sum
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Softmax
    softmax = exp_x / sum_exp
    
    # Sigmoid
    sigmoid = 1.0 / (1.0 + tl.exp(-softmax))
    
    # Store
    tl.store(out_ptr + x_offsets, sigmoid, mask=mask)


def triton_softmax_sigmoid(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    B, C, D, H, W = x.shape
    out = torch.empty_like(x)
    
    stride_b, stride_c, stride_d, stride_h, stride_w = x.stride()
    
    BLOCK_SIZE_C = 64  # Assuming C <= 64, can be adjusted
    
    grid = (B, D, H, W)
    
    softmax_sigmoid_kernel[grid](
        x, out, B, C, D, H, W,
        stride_b, stride_c, stride_d, stride_h, stride_w,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid using a fused Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = triton_softmax_sigmoid(x)
        return x