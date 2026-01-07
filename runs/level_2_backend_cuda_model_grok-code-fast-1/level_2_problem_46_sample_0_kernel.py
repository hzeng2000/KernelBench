import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-conv operations: subtract1, tanh, subtract2
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_post_conv_kernel(const float* x, float* out, float s1, float s2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] - s1;
        val = tanhf(val);
        val = val - s2;
        out[idx] = val;
    }
}

torch::Tensor fused_post_conv_cuda(torch::Tensor x, float s1, float s2) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_post_conv_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), s1, s2, size);

    return out;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor x, float s1, float s2);"
)

# Compile the inline CUDA code for fused post-conv operations
fused_post_conv = load_inline(
    name="fused_post_conv",
    cpp_sources=fused_post_conv_cpp_source,
    cuda_sources=fused_post_conv_source,
    functions=["fused_post_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, fused subtraction-tanh-subtraction, and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_post_conv.fused_post_conv_cuda(x, self.subtract1_value, self.subtract2_value)
        x = self.avgpool(x)
        return x