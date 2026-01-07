import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused post-processing: Mish + Add + Hardtanh + Scale
fused_post_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_post_kernel(const float* input, float* output, float add_val, float scale_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Mish: x * tanh(softplus(x))
        float softplus = logf(1.0f + expf(x));
        float mish = x * tanhf(softplus);
        // Add
        mish += add_val;
        // Hardtanh: clamp to [-1, 1]
        mish = max(-1.0f, min(1.0f, mish));
        // Scale
        mish *= scale_val;
        output[idx] = mish;
    }
}

torch::Tensor fused_post_cuda(torch::Tensor input, float add_val, float scale_val) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_post_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), add_val, scale_val, size);

    return output;
}
"""

fused_post_cpp_source = (
    "torch::Tensor fused_post_cuda(torch::Tensor input, float add_val, float scale_val);"
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
    Optimized Model that performs a transposed convolution and applies a fused custom CUDA kernel for Mish activation, addition, Hardtanh activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.fused_post = fused_post

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_post.fused_post_cuda(x, self.add_value, self.scale)
        return x