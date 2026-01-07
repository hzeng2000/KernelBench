import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused layer normalization, GELU, and scaling
layer_norm_gelu_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void layer_norm_gelu_scale_kernel(const float* __restrict__ input, float* __restrict__ output, int batch, int channels, int depth, int height, int width, float eps, float scaling_factor) {
    int idx = blockIdx.x;
    int b = idx / (channels * depth * height);
    int rem = idx % (channels * depth * height);
    int c = rem / (depth * height);
    rem = rem % (depth * height);
    int d = rem / height;
    int h = rem % height;
    int offset = ((b * channels + c) * depth + d) * height * width + h * width;
    int tid = threadIdx.x;
    float x = input[offset + tid];
    __shared__ float s_data[64];
    s_data[tid] = x;
    __syncthreads();
    // Reduce for sum (mean)
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float mean = s_data[0] / width;
    // Compute variance
    s_data[tid] = (x - mean) * (x - mean);
    __syncthreads();
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float var = s_data[0] / width;
    float norm = (x - mean) / sqrtf(var + eps);
    float gelu_arg = sqrtf(2.0f / M_PI) * (norm + 0.044715f * norm * norm * norm);
    float gelu = 0.5f * norm * (1.0f + tanhf(gelu_arg));
    output[offset + tid] = gelu * scaling_factor;
}

torch::Tensor layer_norm_gelu_scale_cuda(torch::Tensor input, float eps, float scaling_factor) {
    auto sizes = input.sizes();
    int batch = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    auto output = torch::zeros_like(input);
    int num_blocks = batch * channels * depth * height;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(width, 1, 1);
    layer_norm_gelu_scale_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch, channels, depth, height, width, eps, scaling_factor);
    return output;
}
"""

layer_norm_gelu_scale_cpp_source = (
    "torch::Tensor layer_norm_gelu_scale_cuda(torch::Tensor input, float eps, float scaling_factor);"
)

# Compile the inline CUDA code for fused layer normalization, GELU, and scaling
layer_norm_gelu_scale = load_inline(
    name="layer_norm_gelu_scale",
    cpp_sources=layer_norm_gelu_scale_cpp_source,
    cuda_sources=layer_norm_gelu_scale_source,
    functions=["layer_norm_gelu_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, fused layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm_gelu_scale = layer_norm_gelu_scale
        self.eps = eps
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        x = self.layer_norm_gelu_scale.layer_norm_gelu_scale_cuda(x, self.eps, self.scaling_factor)
        return x