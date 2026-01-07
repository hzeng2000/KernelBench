import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused 3D convolution + activations + bias
fused_conv3d_activations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 8

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_conv3d_activations_bias_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int depth, int height, int width, int kernel_size,
    int out_depth, int out_height, int out_width) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (out_x < out_width && out_y < out_height && out_z < out_depth) {
        int batch_idx = blockIdx.w / out_channels;
        int out_channel = blockIdx.w % out_channels;
        
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int kz = 0; kz < kernel_size; ++kz) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_z = out_z - half_kernel + kz;
                        int in_y = out_y - half_kernel + ky;
                        int in_x = out_x - half_kernel + kx;
                        
                        if (in_z >= 0 && in_z < depth && in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            int input_idx = ((batch_idx * in_channels + in_ch) * depth + in_z) * height * width + in_y * width + in_x;
                            int weight_idx = ((out_channel * in_channels + in_ch) * kernel_size + kz) * kernel_size * kernel_size + ky * kernel_size + kx;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Apply activations: ReLU -> LeakyReLU -> GELU -> Sigmoid
        float val = sum;
        
        // ReLU
        val = fmaxf(0.0f, val);
        
        // LeakyReLU
        val = val > 0 ? val : 0.01f * val;
        
        // GELU
        val = gelu(val);
        
        // Sigmoid
        val = sigmoid(val);
        
        // Add bias
        val += bias[out_channel];
        
        int output_idx = ((batch_idx * out_channels + out_channel) * out_depth + out_z) * out_height * out_width + out_y * out_width + out_x;
        output[output_idx] = val;
    }
}

torch::Tensor fused_conv3d_activations_bias_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto out_channels = weight.size(0);
    
    int half_kernel = kernel_size / 2;
    int out_depth = depth - kernel_size + 1 + 2 * half_kernel;
    int out_height = height - kernel_size + 1 + 2 * half_kernel;
    int out_width = width - kernel_size + 1 + 2 * half_kernel;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        (out_depth + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );
    
    fused_conv3d_activations_bias_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        depth, height, width, kernel_size,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

fused_conv3d_activations_cpp_source = (
    "torch::Tensor fused_conv3d_activations_bias_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size);"
)

# Compile the inline CUDA code
fused_conv3d_activations = load_inline(
    name="fused_conv3d_activations",
    cpp_sources=fused_conv3d_activations_cpp_source,
    cuda_sources=fused_conv3d_activations_source,
    functions=["fused_conv3d_activations_bias_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.kernel_size = kernel_size
        self.fused_conv3d_activations = fused_conv3d_activations

    def forward(self, x):
        return self.fused_conv3d_activations.fused_conv3d_activations_bias_cuda(
            x, self.weight, self.bias, self.kernel_size
        )