import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + subtract + mish
fused_conv_sub_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16

__device__ float mish_activation(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_conv_sub_mish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_height, int out_width,
    float subtract1, float subtract2) {
    
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_c = blockIdx.z;
    int b = blockIdx.w;
    
    if (out_y < out_height && out_x < out_width) {
        float sum = 0.0f;
        
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    
                    if (in_y < height && in_x < width) {
                        int input_idx = ((b * in_channels + in_c) * height + in_y) * width + in_x;
                        int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[out_c];
        }
        
        sum -= subtract1;
        sum -= subtract2;
        
        int out_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
        output[out_idx] = mish_activation(sum);
    }
}

torch::Tensor fused_conv_sub_mish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract1, float subtract2) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               input.options());
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_width + TILE_SIZE - 1) / TILE_SIZE,
              (out_height + TILE_SIZE - 1) / TILE_SIZE,
              out_channels,
              batch_size);
    
    fused_conv_sub_mish_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width,
        subtract1, subtract2);
    
    return output;
}
"""

fused_conv_sub_mish_cpp_source = (
    "torch::Tensor fused_conv_sub_mish_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "float subtract1, float subtract2);"
)

# Compile the inline CUDA code
fused_conv_sub_mish = load_inline(
    name="fused_conv_sub_mish",
    cpp_sources=fused_conv_sub_mish_cpp_source,
    cuda_sources=fused_conv_sub_mish_source,
    functions=["fused_conv_sub_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that fuses convolution, subtract operations, and Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.fused_op = fused_conv_sub_mish

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return self.fused_op.fused_conv_sub_mish_cuda(
            x, weight, bias, self.subtract_value_1, self.subtract_value_2
        )