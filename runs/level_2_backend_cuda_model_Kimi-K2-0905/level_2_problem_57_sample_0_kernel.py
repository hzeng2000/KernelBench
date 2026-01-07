import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + relu + hardswish
fused_conv_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void fused_conv_relu_hardswish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_height, int out_width) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    if (out_x < out_width && out_y < out_height && out_c < out_channels) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int in_c = 0; in_c < in_channels; ++in_c) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_y = out_y + ky;
                        int in_x = out_x + kx;
                        if (in_y < height && in_x < width) {
                            int in_idx = ((b * in_channels + in_c) * height + in_y) * width + in_x;
                            int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                            sum += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            int out_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
            sum += bias[out_c];
            // ReLU
            sum = fmaxf(sum, 0.0f);
            // HardSwish
            float hardswish = fminf(fmaxf((sum + 3.0f) / 6.0f, 0.0f), 1.0f);
            output[out_idx] = sum * hardswish;
            sum = 0.0f;
        }
    }
}

torch::Tensor fused_conv_relu_hardswish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);

    const int out_height = height - kernel_size + 1;
    const int out_width = width - kernel_size + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((out_width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (out_height + TILE_WIDTH - 1) / TILE_WIDTH,
                 out_channels);

    fused_conv_relu_hardswish_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width);

    return output;
}
"""

fused_conv_relu_hardswish_cpp_source = (
    "torch::Tensor fused_conv_relu_hardswish_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size);"
)

# Compile the inline CUDA code
fused_conv_relu_hardswish = load_inline(
    name="fused_conv_relu_hardswish",
    cpp_sources=fused_conv_relu_hardswish_cpp_source,
    cuda_sources=fused_conv_relu_hardswish_source,
    functions=["fused_conv_relu_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.kernel_size = kernel_size
        self.fused_op = fused_conv_relu_hardswish

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return self.fused_op.fused_conv_relu_hardswish_cuda(x, weight, bias, self.kernel_size)