import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused 3D conv + leaky_relu + add + clamp + gelu
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 8

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ float leaky_relu(float x, float negative_slope) {
    return x > 0 ? x : x * negative_slope;
}

__global__ void fused_conv_activation_kernel(
    const float* input, const float* weight, const float* bias,
    const float* sum_tensor, float* output,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding,
    float negative_slope) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (out_x < width && out_y < height && out_z < depth) {
        int batch_idx = blockIdx.w / out_channels;
        int out_channel = blockIdx.w % out_channels;
        
        float sum = 0.0f;
        
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int kz = 0; kz < kernel_size; ++kz) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_z = out_z * stride - padding + kz;
                        int in_y = out_y * stride - padding + ky;
                        int in_x = out_x * stride - padding + kx;
                        
                        if (in_z >= 0 && in_z < depth && in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            int input_idx = ((batch_idx * in_channels + in_ch) * depth + in_z) * height * width + in_y * width + in_x;
                            int weight_idx = ((out_channel * in_channels + in_ch) * kernel_size + kz) * kernel_size * kernel_size + ky * kernel_size + kx;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[out_channel];
        }
        
        // LeakyReLU
        sum = leaky_relu(sum, negative_slope);
        
        // Add sum_tensor
        sum += sum_tensor[out_channel];
        
        // Clamp
        sum = fmaxf(-1.0f, fminf(1.0f, sum));
        
        // GELU
        sum = gelu(sum);
        
        int output_idx = ((batch_idx * out_channels + out_channel) * depth + out_z) * height * width + out_y * width + out_x;
        output[output_idx] = sum;
    }
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor sum_tensor, float negative_slope) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int stride = 1;
    const int padding = 0;
    
    auto output = torch::zeros({batch_size, out_channels, depth - kernel_size + 1, height - kernel_size + 1, width - kernel_size + 1}, input.options());
    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(
        (width - kernel_size + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height - kernel_size + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (depth - kernel_size + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size * out_channels
    );
    
    fused_conv_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        sum_tensor.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth - kernel_size + 1, height - kernel_size + 1, width - kernel_size + 1,
        kernel_size, stride, padding, negative_slope
    );
    
    return output;
}
"""

fused_conv_activation_cpp_source = (
    "torch::Tensor fused_conv_activation_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor sum_tensor, float negative_slope);"
)

# Compile the inline CUDA code
fused_conv_activation = load_inline(
    name="fused_conv_activation",
    cpp_sources=fused_conv_activation_cpp_source,
    cuda_sources=fused_conv_activation_source,
    functions=["fused_conv_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_conv_activation = fused_conv_activation

    def forward(self, x):
        return self.fused_conv_activation.fused_conv_activation_cuda(
            x, self.conv.weight, self.conv.bias, self.sum_tensor, 0.2
        )