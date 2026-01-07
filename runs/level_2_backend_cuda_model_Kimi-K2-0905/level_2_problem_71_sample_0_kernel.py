import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + div + leaky_relu
fused_conv_div_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_conv_div_leakyrelu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int divisor, float negative_slope) {

    int out_h = (height - kernel_size) + 1;
    int out_w = (width - kernel_size) + 1;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    if (row < out_h && col < out_w && bz < batch_size) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float sum = 0.0f;
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int in_row = row + kh;
                        int in_col = col + kw;
                        int in_idx = ((bz * in_channels + ic) * height + in_row) * width + in_col;
                        int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
            if (bias != nullptr) {
                sum += bias[oc];
            }
            sum = sum / divisor;
            if (sum < 0.0f) {
                sum = sum * negative_slope;
            }
            int out_idx = ((bz * out_channels + oc) * out_h + row) * out_w + col;
            output[out_idx] = sum;
        }
    }
}

torch::Tensor fused_conv_div_leakyrelu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int divisor, float negative_slope) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);

    const int out_h = height - kernel_size + 1;
    const int out_w = width - kernel_size + 1;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((out_w + TILE_SIZE - 1) / TILE_SIZE,
                 (out_h + TILE_SIZE - 1) / TILE_SIZE,
                 batch_size);

    fused_conv_div_leakyrelu_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, kernel_size, divisor, negative_slope);

    return output;
}
"""

fused_conv_div_leakyrelu_cpp_source = (
    "torch::Tensor fused_conv_div_leakyrelu_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int divisor, float negative_slope);"
)

# Compile the inline CUDA code for fused operations
fused_conv_div_leakyrelu = load_inline(
    name="fused_conv_div_leakyrelu",
    cpp_sources=fused_conv_div_leakyrelu_cpp_source,
    cuda_sources=fused_conv_div_leakyrelu_source,
    functions=["fused_conv_div_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_op = fused_conv_div_leakyrelu

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return self.fused_op.fused_conv_div_leakyrelu_cuda(x, weight, bias, self.divisor, 0.01)