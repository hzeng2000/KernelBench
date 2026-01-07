import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose2D + add + min + GELU + multiply
fused_conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 16

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_size, int stride, float add_value, float multiply_value) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    if (out_x >= out_width || out_y >= out_height || out_c >= out_channels) return;

    int batch = blockIdx.w;
    float sum = 0.0f;

    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = (out_y - ky) / stride;
                int in_x = (out_x - kx) / stride;

                if ((out_y - ky) % stride == 0 && (out_x - kx) % stride == 0 &&
                    in_y >= 0 && in_x >= 0 && in_y < in_height && in_x < in_width) {

                    int in_idx = ((batch * in_channels + in_c) * in_height + in_y) * in_width + in_x;
                    int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[out_c];
    }

    sum += add_value;
    sum = fminf(sum, 0.0f);
    sum = gelu(sum);
    sum *= multiply_value;

    int out_idx = ((batch * out_channels + out_c) * out_height + out_y) * out_width + out_x;
    output[out_idx] = sum;
}

torch::Tensor fused_conv_transpose_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, float add_value, float multiply_value) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);
    const auto out_channels = weight.size(0);

    const auto out_height = (in_height - 1) * stride + kernel_size;
    const auto out_width = (in_width - 1) * stride + kernel_size;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
              out_channels);
    grid.x = (out_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid.y = (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid.z = out_channels;

    for (int b = 0; b < batch_size; ++b) {
        fused_conv_transpose_kernel<<<grid, block>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(), batch_size, in_channels, out_channels,
            in_height, in_width, out_height, out_width,
            kernel_size, stride, add_value, multiply_value);
    }

    return output;
}
"""

fused_conv_transpose_cpp_source = (
    "torch::Tensor fused_conv_transpose_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int kernel_size, int stride, float add_value, float multiply_value);"
)

fused_conv_transpose = load_inline(
    name="fused_conv_transpose",
    cpp_sources=fused_conv_transpose_cpp_source,
    cuda_sources=fused_conv_transpose_source,
    functions=["fused_conv_transpose_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_op = fused_conv_transpose

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return self.fused_op.fused_conv_transpose_cuda(
            x, weight, bias, self.conv_transpose.kernel_size[0],
            self.conv_transpose.stride[0], self.add_value, self.multiply_value
        )