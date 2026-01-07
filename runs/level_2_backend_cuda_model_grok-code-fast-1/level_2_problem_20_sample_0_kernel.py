import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations after ConvTranspose3d
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* y, const float* bias, float* out, int size, int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / spatial) % C;
        float yy = y[idx];
        float bb = bias[c];
        out[idx] = (2.0f * yy + bb) * yy + yy;
    }
}

torch::Tensor fused_cuda(torch::Tensor y, torch::Tensor bias) {
    auto out = torch::empty_like(y);
    int size = y.numel();
    int C = bias.size(0);
    int spatial = y.size(2) * y.size(3) * y.size(4);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_kernel<<<num_blocks, block_size>>>(y.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), size, C, spatial);
    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_cuda(torch::Tensor y, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused operations
fused = load_inline(
    name="fused",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by fused custom CUDA operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused = fused

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused.fused_cuda(x, self.bias)
        return x