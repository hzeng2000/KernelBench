import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2D + Mish + Mish
conv_mish_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void conv_mish_mish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_height, int out_width) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    
    if (out_x < out_width && out_y < out_height && out_c < out_channels) {
        float sum = 0.0f;
        
        for (int b = 0; b < batch_size; b++) {
            for (int in_c = 0; in_c < in_channels; in_c++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_y = out_y + ky;
                        int in_x = out_x + kx;
                        
                        if (in_y < height && in_x < width) {
                            int input_idx = b * in_channels * height * width +
                                          in_c * height * width +
                                          in_y * width + in_x;
                            int weight_idx = out_c * in_channels * kernel_size * kernel_size +
                                           in_c * kernel_size * kernel_size +
                                           ky * kernel_size + kx;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            if (bias != nullptr) {
                sum += bias[out_c];
            }
            
            // First Mish activation
            float x1 = sum;
            float softplus1 = logf(1.0f + expf(x1));
            float tanh1 = tanhf(softplus1);
            float mish1 = x1 * tanh1;
            
            // Second Mish activation
            float x2 = mish1;
            float softplus2 = logf(1.0f + expf(x2));
            float tanh2 = tanhf(softplus2);
            float mish2 = x2 * tanh2;
            
            int output_idx = b * out_channels * out_height * out_width +
                           out_c * out_height * out_width +
                           out_y * out_width + out_x;
            output[output_idx] = mish2;
        }
    }
}

torch::Tensor conv_mish_mish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = height - kernel_size + 1;
    const int out_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((out_width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (out_height + TILE_WIDTH - 1) / TILE_WIDTH,
                 out_channels);
    
    conv_mish_mish_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width);
    
    return output;
}
"""

conv_mish_mish_cpp_source = (
    "torch::Tensor conv_mish_mish_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int stride, int padding);"
)

# Compile the inline CUDA code
conv_mish_mish = load_inline(
    name="conv_mish_mish",
    cpp_sources=conv_mish_mish_cpp_source,
    cuda_sources=conv_mish_mish_source,
    functions=["conv_mish_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_mish_mish = conv_mish_mish

    def forward(self, x):
        return self.conv_mish_mish.conv_mish_mish_cuda(
            x, self.conv.weight, self.conv.bias, 1, 0)