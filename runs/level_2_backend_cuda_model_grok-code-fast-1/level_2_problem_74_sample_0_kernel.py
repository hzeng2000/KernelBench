import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-conv operations: LeakyReLU -> element-wise mul -> LeakyReLU
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* input, const float* multiplier, float* output, int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * depth * height * width;
    if (idx >= total) return;
    
    int b = idx / (channels * depth * height * width);
    int rem = idx % (channels * depth * height * width);
    int c = rem / (depth * height * width);
    rem %= (depth * height * width);
    int z = rem / (height * width);
    rem %= (height * width);
    int y = rem / width;
    int x = rem % width;
    
    float val = input[idx];
    val = val > 0 ? val : 0.2f * val;
    val *= multiplier[c];
    val = val > 0 ? val : 0.2f * val;
    output[idx] = val;
}

torch::Tensor fused_post_conv_cuda(torch::Tensor input, torch::Tensor multiplier) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto out = torch::zeros_like(input);
    
    int total = input.numel();
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    
    fused_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), multiplier.data_ptr<float>(), out.data_ptr<float>(), batch, channels, depth, height, width);
    
    return out;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor input, torch::Tensor multiplier);"
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
    Optimized Model that performs a 3D transposed convolution, applies fused LeakyReLU + multiplication + LeakyReLU, and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_op = fused_post_conv
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_post_conv_cuda(x, self.multiplier)
        x = self.max_pool(x)
        return x