import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def logsumexp_relu_kernel(
    x_ptr,
    y_ptr,
    B, C, D, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    stride_y_b, stride_y_d, stride_y_h, stride_y_w,
):
    pid = tl.program_id(0)
    w = pid % W
    h = (pid // W) % H
    d = (pid // (W * H)) % D
    b = pid // (W * H * D)
    base_offset = b * stride_b + d * stride_d + h * stride_h + w * stride_w
    offsets = base_offset + tl.arange(0, C) * stride_c
    x_vals = tl.load(x_ptr + offsets)
    max_val = tl.reduce(x_vals, tl.maximum, 0)
    exp_vals = tl.exp(x_vals - max_val)
    sum_exp = tl.reduce(exp_vals, tl.add, 0)
    logsumexp_val = tl.log(sum_exp) + max_val
    relu_val = tl.maximum(logsumexp_val, 0.0)
    y_offset = b * stride_y_b + d * stride_y_d + h * stride_y_h + w * stride_y_w
    tl.store(y_ptr + y_offset, relu_val)


def triton_logsumexp_relu(x: torch.Tensor):
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    B, C, D, H, W = x.shape
    y = torch.empty(B, 1, D, H, W, device=x.device, dtype=x.dtype)
    stride_b = C * D * H * W
    stride_c = D * H * W
    stride_d = H * W
    stride_h = W
    stride_w = 1
    stride_y_b = 1 * D * H * W
    stride_y_d = H * W
    stride_y_h = W
    stride_y_w = 1
    total_spatial = B * D * H * W
    grid = (total_spatial,)
    logsumexp_relu_kernel[grid](
        x, y, B, C, D, H, W,
        stride_b, stride_c, stride_d, stride_h, stride_w,
        stride_y_b, stride_y_d, stride_y_h, stride_y_w
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        x = triton_logsumexp_relu(x)
        return x