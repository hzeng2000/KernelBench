import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-conv operations: LeakyReLU, add, clamp, GELU
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_post_conv_kernel(const float* x, const float* sum_tensor, float* out, int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * depth * height * width;
    if (idx >= total) return;

    int c = (idx / (depth * height * width)) % channels;

    float val = x[idx];

    // LeakyReLU
    val = val > 0.0f ? val : 0.2f * val;

    // Add sum_tensor[c]
    val += sum_tensor[c];

    // Clamp
    val = max(-1.0f, min(1.0f, val));

    // GELU
    float sqrt_2_pi = sqrtf(2.0f / M_PI);
    float x3 = val * val * val;
    float tanh_arg = sqrt_2_pi * (val + 0.044715f * x3);
    val = 0.5f * val * (1.0f + tanhf(tanh_arg));

    out[idx] = val;
}

torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor sum_tensor) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    auto out = torch::zeros_like(x);

    int total = x.numel();
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    fused_post_conv_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), sum_tensor.data_ptr<float>(), out.data_ptr<float>(), batch, channels, depth, height, width);

    return out;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor sum_tensor);"
)

# Compile the inline CUDA code for fused post-conv
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
    Optimized Model that performs a 3D convolution, then applies fused LeakyReLU, addition, clamp, and GELU in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_post_conv.fused_post_conv_cuda(x, self.sum_tensor)
        return x