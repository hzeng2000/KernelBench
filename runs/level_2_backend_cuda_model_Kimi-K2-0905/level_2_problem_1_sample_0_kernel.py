import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + relu + add_bias
fused_conv_relu_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void conv_relu_bias_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_height, int out_width) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    
    if (out_x < out_width && out_y < out_height && out_c < out_channels) {
        float sum = 0.0f;
        int pad = kernel_size / 2;
        
        for (int n = 0; n < batch_size; n++) {
            for (int c = 0; c < in_channels; c++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_y = out_y - pad + ky;
                        int in_x = out_x - pad + kx;
                        
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            int in_idx = n * in_channels * height * width + c * height * width + in_y * width + in_x;
                            int weight_idx = out_c * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + ky * kernel_size + kx;
                            sum += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            int out_idx = n * out_channels * out_height * out_width + out_c * out_height * out_width + out_y * out_width + out_x;
            float val = sum + bias[out_c];
            output[out_idx] = fmaxf(0.0f, val);
        }
    }
}

torch::Tensor fused_conv_relu_bias_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = height;
    int out_width = width;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((out_width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (out_height + TILE_WIDTH - 1) / TILE_WIDTH,
                 out_channels);
    
    conv_relu_bias_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width);
    
    return output;
}
"""

fused_conv_relu_bias_cpp_source = (
    "torch::Tensor fused_conv_relu_bias_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code for fused operations
fused_conv_relu_bias = load_inline(
    name="fused_conv_relu_bias",
    cpp_sources=fused_conv_relu_bias_cpp_source,
    cuda_sources=fused_conv_relu_bias_source,
    functions=["fused_conv_relu_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_conv_relu_bias

    def forward(self, x):
        return self.fused_op.fused_conv_relu_bias_cuda(x, self.conv.weight, self.bias.view(-1))