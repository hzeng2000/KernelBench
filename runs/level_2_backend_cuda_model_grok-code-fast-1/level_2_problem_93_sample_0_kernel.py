import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-conv operations: add scalar, min with 0, GELU, multiply scalar
fused_post_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_post_conv_kernel(const float* input, float* out, int size, float add_value, float multiply_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] + add_value;
        val = fminf(val, 0.0f);
        val = 0.5f * val * (1.0f + erff(val / sqrtf(2.0f)));
        val = val * multiply_value;
        out[idx] = val;
    }
}

torch::Tensor fused_post_conv_cuda(torch::Tensor input, float add_value, float multiply_value) {
    auto size = input.numel();
    auto out = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_post_conv_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), out.data_ptr<float>(), size, add_value, multiply_value);

    return out;
}
"""

fused_post_conv_cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor input, float add_value, float multiply_value);"
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
    Optimized Model that performs a transposed convolution, then applies fused custom CUDA operator for add, min with 0, GELU, and multiply.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_post_conv = fused_post_conv

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_post_conv.fused_post_conv_cuda(x, self.add_value, self.multiply_value)
        return x