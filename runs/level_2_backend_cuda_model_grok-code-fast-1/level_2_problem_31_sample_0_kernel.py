import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused min, add bias, and multiply scaling
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* x, float constant, const float* bias, float scale, float* out, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * H * W;
    if (idx < total_size) {
        int c = (idx / (H * W)) % C;
        float val = x[idx];
        val = min(val, constant);
        val += bias[c];
        val *= scale;
        out[idx] = val;
    }
}

torch::Tensor fused_cuda(torch::Tensor x, float constant, torch::Tensor bias, float scale) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto out = torch::zeros_like(x);
    int size = N * C * H * W;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    fused_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), constant, bias.data_ptr<float>(), scale, out.data_ptr<float>(), N, C, H, W);
    return out;
}
"""

fused_cpp_source = (
    "torch::Tensor fused_cuda(torch::Tensor x, float constant, torch::Tensor bias, float scale);"
)

# Compile the inline CUDA code for fused operation
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
    Optimized model that performs a convolution, then fuses the minimum with a constant, adds a bias term, and multiplies by a scaling factor into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused = fused

    def forward(self, x):
        x = self.conv(x)
        x = self.fused.fused_cuda(x, self.constant_value, self.bias, self.scaling_factor)
        return x