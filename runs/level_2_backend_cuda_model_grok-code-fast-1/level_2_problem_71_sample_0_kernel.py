import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused division and LeakyReLU
fused_div_leaky_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_div_leaky_kernel(const float* x, float* out, int size, float divisor, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] / divisor;
        out[idx] = val > 0.0f ? val : negative_slope * val;
    }
}

torch::Tensor fused_div_leaky_cuda(torch::Tensor x, float divisor, float negative_slope) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_div_leaky_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, divisor, negative_slope);

    return out;
}
"""

fused_div_leaky_cpp_source = (
    "torch::Tensor fused_div_leaky_cuda(torch::Tensor x, float divisor, float negative_slope);"
)

# Compile the inline CUDA code for fused division and LeakyReLU
fused_div_leaky = load_inline(
    name="fused_div_leaky",
    cpp_sources=fused_div_leaky_cpp_source,
    cuda_sources=fused_div_leaky_source,
    functions=["fused_div_leaky_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, then applies a fused division and LeakyReLU in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_div_leaky = fused_div_leaky

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_div_leaky.fused_div_leaky_cuda(x, self.divisor, 0.01)
        return x