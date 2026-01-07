import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation followed by bias addition
swish_add_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_add_bias_kernel(const float* x, const float* bias, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + __expf(-val));
        out[idx] = val * sig + bias[idx];
    }
}

torch::Tensor swish_add_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_add_bias_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

swish_add_bias_cpp_source = (
    "torch::Tensor swish_add_bias_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for Swish + bias addition
swish_add_bias = load_inline(
    name="swish_add_bias",
    cpp_sources=swish_add_bias_cpp_source,
    cuda_sources=swish_add_bias_source,
    functions=["swish_add_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    Optimized by fusing Swish activation and bias addition into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.swish_add_bias = swish_add_bias

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.swish_add_bias.swish_add_bias_cuda(x, self.bias)
        x = self.group_norm(x)
        return x