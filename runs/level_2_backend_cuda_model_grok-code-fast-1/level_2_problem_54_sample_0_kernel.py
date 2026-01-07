import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-processing: multiply by multiplier, LeakyReLU, GELU
fused_post_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_post_kernel(const float* x, const float* multiplier, float* out, int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * height * width;
    if (idx >= total) return;

    int w = idx % width;
    idx /= width;
    int h = idx % height;
    idx /= height;
    int c = idx % channels;
    idx /= channels;
    int b = idx;

    int x_offset = b * channels * height * width + c * height * width + h * width + w;
    float val = x[x_offset];
    float mult = multiplier[c];
    val = val * mult;

    // LeakyReLU with negative_slope=0.01
    val = (val > 0.0f) ? val : 0.01f * val;

    // GELU: 0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
    float x_val = val;
    float x3 = x_val * x_val * x_val;
    float inner = sqrtf(2.0f / M_PI) * (x_val + 0.044715f * x3);
    val = 0.5f * x_val * (1.0f + tanhf(inner));

    out[x_offset] = val;
}

torch::Tensor fused_post_cuda(torch::Tensor x, torch::Tensor multiplier) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto out = torch::zeros_like(x);

    int total = batch * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    fused_post_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), multiplier.data_ptr<float>(), out.data_ptr<float>(), batch, channels, height, width);

    return out;
}
"""

fused_post_cpp_source = (
    "torch::Tensor fused_post_cuda(torch::Tensor x, torch::Tensor multiplier);"
)

# Compile the inline CUDA code for fused post-processing
fused_post = load_inline(
    name="fused_post",
    cpp_sources=fused_post_cpp_source,
    cuda_sources=fused_post_source,
    functions=["fused_post_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, then fused multiply by learnable scalar, LeakyReLU, and GELU in a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.fused_post = fused_post

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_post.fused_post_cuda(x, self.multiplier)
        return x