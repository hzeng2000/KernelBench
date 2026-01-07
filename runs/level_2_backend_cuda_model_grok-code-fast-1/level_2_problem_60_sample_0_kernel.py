import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Swish and HardSwish activations
activations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val / (1.0f + expf(-val));
    }
}

__global__ void hardswish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float clamped = fmaxf(fminf(val + 3.0f, 6.0f), 0.0f);
        out[idx] = val * clamped / 6.0f;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}

torch::Tensor hardswish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardswish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

activations_cpp_source = (
    "torch::Tensor swish_cuda(torch::Tensor x);"
    "torch::Tensor hardswish_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for activations
activations = load_inline(
    name="activations",
    cpp_sources=activations_cpp_source,
    cuda_sources=activations_source,
    functions=["swish_cuda", "hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, applies custom CUDA Swish activation, 
    group normalization, and then custom CUDA HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.swish = activations
        self.hardswish = activations

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.swish.swish_cuda(x)
        x = self.group_norm(x)
        x = self.hardswish.hardswish_cuda(x)
        return x