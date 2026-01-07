import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + min + add + multiply
fused_conv_min_add_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_conv_min_add_mul_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, float constant_value,
    float scaling_factor) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    int b = blockIdx.w;
    
    if (out_x >= width || out_y >= height || out_c >= out_channels) return;
    
    float sum = 0.0f;
    int pad = kernel_size / 2;
    
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = out_x - pad + kx;
                int in_y = out_y - pad + ky;
                
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    int input_idx = ((b * in_channels + in_c) * height + in_y) * width + in_x;
                    int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    int output_idx = ((b * out_channels + out_c) * height + out_y) * width + out_x;
    float result = sum + bias[out_c];
    result = fminf(result, constant_value);
    result = result * scaling_factor;
    output[output_idx] = result;
}

torch::Tensor fused_conv_min_add_mul_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float constant_value, float scaling_factor) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE,
                   (height + TILE_SIZE - 1) / TILE_SIZE,
                   out_channels,
                   batch_size);
    
    fused_conv_min_add_mul_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width,
        kernel_size, constant_value, scaling_factor);
    
    return output;
}
"""

fused_conv_min_add_mul_cpp_source = (
    "torch::Tensor fused_conv_min_add_mul_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "float constant_value, float scaling_factor);"
)

# Compile the inline CUDA code
fused_conv_min_add_mul = load_inline(
    name="fused_conv_min_add_mul",
    cpp_sources=fused_conv_min_add_mul_cpp_source,
    cuda_sources=fused_conv_min_add_mul_source,
    functions=["fused_conv_min_add_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_op = fused_conv_min_add_mul

    def forward(self, x):
        return self.fused_op.fused_conv_min_add_mul_cuda(
            x, self.conv.weight, self.bias, self.constant_value, self.scaling_factor
        )