import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction along channels followed by two tanh operations
min_tanh_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void min_tanh_tanh_kernel(const float* input, float* output, int batch, int channels, int height, int width) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    if (b >= batch || h >= height || w >= width) return;

    int idx_base = ((b * channels * height + h) * width + w);
    float min_val = input[idx_base];
    for (int c = 1; c < channels; ++c) {
        int idx = idx_base + c * height * width;
        if (input[idx] < min_val) min_val = input[idx];
    }
    float val = tanhf(tanhf(min_val));
    int out_idx = ((b * 1 * height + h) * width + w);
    output[out_idx] = val;
}

torch::Tensor min_tanh_tanh_cuda(torch::Tensor input) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros({batch, 1, height, width}, input.options());

    dim3 blocks(batch, height, width);
    min_tanh_tanh_kernel<<<blocks, 1>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch, channels, height, width);

    return output;
}
"""

min_tanh_tanh_cpp_source = (
    "torch::Tensor min_tanh_tanh_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for min + tanh + tanh
min_tanh_tanh = load_inline(
    name="min_tanh_tanh",
    cpp_sources=min_tanh_tanh_cpp_source,
    cuda_sources=min_tanh_tanh_source,
    functions=["min_tanh_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, applies minimum operation, Tanh, and another Tanh using a fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.min_tanh_tanh = min_tanh_tanh

    def forward(self, x):
        x = self.conv(x)
        x = self.min_tanh_tanh.min_tanh_tanh_cuda(x)
        return x