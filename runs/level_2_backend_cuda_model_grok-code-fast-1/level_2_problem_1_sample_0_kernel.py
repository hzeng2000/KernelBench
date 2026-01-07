import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU followed by bias addition
relu_add_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_add_bias_kernel(const float* x, const float* bias, float* out, int size, int channels, int hw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / hw) % channels;
        float val = x[idx];
        out[idx] = max(0.0f, val) + bias[c];
    }
}

torch::Tensor relu_add_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    int channels = x.size(1);
    int hw = x.size(2) * x.size(3);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_add_bias_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), size, channels, hw);

    return out;
}
"""

relu_add_bias_cpp_source = (
    "torch::Tensor relu_add_bias_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for ReLU + bias addition
relu_add_bias = load_inline(
    name="relu_add_bias",
    cpp_sources=relu_add_bias_cpp_source,
    cuda_sources=relu_add_bias_source,
    functions=["relu_add_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies fused ReLU and bias addition using custom CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.relu_add_bias = relu_add_bias

    def forward(self, x):
        x = self.conv(x)
        x = self.relu_add_bias.relu_add_bias_cuda(x, self.bias)
        return x