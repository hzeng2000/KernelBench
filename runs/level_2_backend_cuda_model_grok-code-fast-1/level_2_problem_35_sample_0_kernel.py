import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for subtract_hardswish and mish
custom_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void subtract_hardswish_kernel(const float* input, float* output, float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - val;
        // hardswish: x * relu6(x + 3) / 6
        float relu6_val = min(max(x + 3.0f, 0.0f), 6.0f);
        output[idx] = x * relu6_val / 6.0f;
    }
}

__global__ void mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // mish: x * tanh(softplus(x))
        float softplus = logf(1.0f + expf(x));
        output[idx] = x * tanhf(softplus);
    }
}

torch::Tensor subtract_hardswish_cuda(torch::Tensor input, float val) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtract_hardswish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), val, size);

    return output;
}

torch::Tensor mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

custom_ops_cpp_source = (
    "torch::Tensor subtract_hardswish_cuda(torch::Tensor input, float val);\n"
    "torch::Tensor mish_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for custom operations
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=custom_ops_cpp_source,
    cuda_sources=custom_ops_source,
    functions=["subtract_hardswish_cuda", "mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, subtracts a value with fused HardSwish, MaxPool, and Mish activation functions using custom CUDA operators.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.custom_ops = custom_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.custom_ops.subtract_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        x = self.custom_ops.mish_cuda(x)
        return x