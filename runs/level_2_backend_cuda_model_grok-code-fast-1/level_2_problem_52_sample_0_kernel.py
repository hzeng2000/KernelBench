import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the activation function
activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void activation_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float sp = logf(1.0f + expf(val));  // softplus
        float t = tanhf(sp);
        out[idx] = val * t;
    }
}

torch::Tensor activation_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    activation_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

activation_cpp_source = (
    "torch::Tensor activation_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for activation
activation = load_inline(
    name="activation",
    cpp_sources=activation_cpp_source,
    cuda_sources=activation_source,
    functions=["activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies custom CUDA activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation.activation_cuda(x)
        x = self.bn(x)
        return x