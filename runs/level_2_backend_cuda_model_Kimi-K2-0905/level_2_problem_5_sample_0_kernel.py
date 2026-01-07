import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d + Bias subtraction + Tanh fusion
conv_transpose_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void conv_transpose_bias_tanh_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, 
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_size, int stride, int padding, int output_padding) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;

    if (out_x >= out_width || out_y >= out_height || n >= batch_size) return;

    float value = 0.0f;
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = (out_y + padding - ky * 1) / stride;
                int in_x = (out_x + padding - kx * 1) / stride;

                if ((out_y + padding - ky) % stride != 0 || (out_x + padding - kx) % stride != 0)
                    continue;
                if (in_y < 0 || in_x < 0 || in_y >= in_height || in_x >= in_width)
                    continue;

                int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                int input_idx = ((n * in_channels + in_c) * in_height + in_y) * in_width + in_x;
                value += input[input_idx] * weight[weight_idx];
            }
        }
    }

    int out_idx = ((n * out_channels + out_c) * out_height + out_y) * out_width + out_x;
    value += bias[out_c];
    output[out_idx] = tanhf(value);
}
"""

conv_transpose_bias_tanh_cpp_source = (
    "torch::Tensor conv_transpose_bias_tanh_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int stride, int padding, int output_padding);"
)

conv_transpose_bias_tanh = load_inline(
    name="conv_transpose_bias_tanh",
    cpp_sources=conv_transpose_bias_tanh_cpp_source,
    cuda_sources=conv_transpose_bias_tanh_source,
    functions=["conv_transpose_bias_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.conv_transpose_bias_tanh = conv_transpose_bias_tanh

    def forward(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        in_height = x.size(2)
        in_width = x.size(3)
        out_channels = self.weight.size(0)
        kernel_size = self.weight.size(2)
        out_height = (in_height - 1) * self.stride - 2 * self.padding + kernel_size + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + kernel_size + self.output_padding

        output = torch.zeros(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

        threads = 16
        dimBlock = dim3(threads, threads, 1)
        dimGrid = dim3(
            (out_width + threads - 1) / threads,
            (out_height + threads - 1) / threads,
            batch_size * out_channels
        );

        conv_transpose_bias_tanh_cuda = self.conv_transpose_bias_tanh.conv_transpose_bias_tanh_cuda
        conv_transpose_bias_tanh_cuda(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding
        )

        return output