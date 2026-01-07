import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-convolution operations
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_post_conv_kernel(const float* input, const float* scaling, const float* bias, float* output, int batch_size, int out_channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * depth * height * width;
    if (idx >= total_elements) return;

    int c = (idx / (depth * height * width)) % out_channels;

    float val = input[idx];
    val *= scaling[c];
    val = tanhf(val);
    val *= bias[c];
    val = 1.0f / (1.0f + expf(-val));  // sigmoid
    output[idx] = val;
}

torch::Tensor fused_post_conv_cuda(torch::Tensor input, torch::Tensor scaling, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto out_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto output = torch::zeros_like(input);

    int total_elements = batch_size * out_channels * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_post_conv_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), scaling.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_channels, depth, height, width);

    return output;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor input, torch::Tensor scaling, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused post-convolution operations
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
    Optimized Model that performs a 3D convolution and then applies fused custom CUDA operations for scaling, tanh, bias multiplication, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv(x)
        return self.fused_post_conv.fused_post_conv_cuda(x, self.scaling_factor, self.bias)