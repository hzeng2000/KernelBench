import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSwish activation
hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardswish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float clamped = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[idx] = x * clamped / 6.0f;
    }
}

torch::Tensor hardswish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardswish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

# Define the custom CUDA kernel for mean pooling across spatial dimensions
mean_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_pool_kernel(const float* input, float* output, int B, int C, int D, int H, int W) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    if (b >= B || c >= C) return;
    float sum = 0.0f;
    int num_elements = D * H * W;
    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int idx = ((b * C + c) * D + d) * H * W + h * W + w;
                sum += input[idx];
            }
        }
    }
    output[b * C + c] = sum / num_elements;
}

torch::Tensor mean_pool_cuda(torch::Tensor input) {
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto output = torch::empty({B, C}, input.options());
    dim3 blocks(B, C);
    mean_pool_kernel<<<blocks, 1>>>(input.data_ptr<float>(), output.data_ptr<float>(), B, C, D, H, W);
    return output;
}
"""

cpp_source = (
    "torch::Tensor hardswish_cuda(torch::Tensor input);\n"
    "torch::Tensor mean_pool_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for HardSwish and mean pooling
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=hardswish_source + mean_pool_source,
    functions=["hardswish_cuda", "mean_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs:
    1. Conv3D (unchanged)
    2. HardSwish activation (custom CUDA)
    3. GroupNorm (unchanged)
    4. Mean pooling across spatial dimensions (custom CUDA)
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.hardswish = custom_ops
        self.mean_pool = custom_ops

    def forward(self, x):
        x = self.conv(x)                             # (B, C, D, H, W)
        x = self.hardswish.hardswish_cuda(x)         # Nonlinear activation
        x = self.group_norm(x)                       # Normalization over channels
        x = self.mean_pool.mean_pool_cuda(x)         # Mean over spatial dims → (B, C)
        return x