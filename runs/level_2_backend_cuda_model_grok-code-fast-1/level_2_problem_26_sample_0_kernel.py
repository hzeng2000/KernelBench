import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused add and HardSwish
fused_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_hardswish_kernel(const float* conv_out, const float* add_input, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y = conv_out[idx] + add_input[idx];
        float relu6 = min(max(y + 3.0f, 0.0f), 6.0f);
        out[idx] = y * relu6 / 6.0f;
    }
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor conv_out, torch::Tensor add_input) {
    auto size = conv_out.numel();
    auto out = torch::zeros_like(conv_out);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hardswish_kernel<<<num_blocks, block_size>>>(conv_out.data_ptr<float>(), add_input.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_add_hardswish_cpp_source = (
    "torch::Tensor fused_add_hardswish_cuda(torch::Tensor conv_out, torch::Tensor add_input);"
)

# Compile the inline CUDA code for fused add and HardSwish
fused_add_hardswish = load_inline(
    name="fused_add_hardswish",
    cpp_sources=fused_add_hardswish_cpp_source,
    cuda_sources=fused_add_hardswish_source,
    functions=["fused_add_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_add_hardswish = fused_add_hardswish

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution, of shape (batch_size, out_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W) after HardSwish activation.
        """
        x = self.conv_transpose(x)
        x = self.fused_add_hardswish.fused_add_hardswish_cuda(x, add_input)
        return x