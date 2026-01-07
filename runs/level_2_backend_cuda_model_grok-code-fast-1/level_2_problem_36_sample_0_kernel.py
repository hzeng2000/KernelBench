import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused min, sum, gelu, and add operations
custom_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void custom_kernel(const float* x, float* out, int B, int C, int H, int W, float bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * W) return;
    int b = idx / W;
    int w = idx % W;
    float sum_val = 0.0f;
    for (int h = 0; h < H; h++) {
        float min_val = 1e10f;
        for (int c = 0; c < C; c++) {
            int index = ((b * C + c) * H + h) * W + w;
            float v = x[index];
            if (v < min_val) min_val = v;
        }
        sum_val += min_val;
    }
    float gelu_val = gelu(sum_val);
    out[b * W + w] = gelu_val + bias;
}

torch::Tensor custom_cuda(torch::Tensor x, torch::Tensor bias) {
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto out = torch::empty({B, 1, 1, W}, x.options());
    int total = B * W;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    custom_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), B, C, H, W, bias.data_ptr<float>()[0]);
    return out;
}
"""

custom_cpp_source = (
    "torch::Tensor custom_cuda(torch::Tensor x, torch::Tensor bias);"
)

# Compile the inline CUDA code for the custom kernel
custom = load_inline(
    name="custom",
    cpp_sources=custom_cpp_source,
    cuda_sources=custom_source,
    functions=["custom_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.custom = custom

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.custom.custom_cuda(x, self.bias)
        return x