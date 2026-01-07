import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + multiply + leaky_relu + gelu
fused_conv_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 16

__device__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_activation_kernel(
    const float* input, const float* weight, const float* bias, const float* multiplier,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_height, int out_width) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    
    if (out_x < out_width && out_y < out_height && out_c < out_channels) {
        float sum = 0.0f;
        
        for (int n = 0; n < batch_size; n++) {
            for (int c = 0; c < in_channels; c++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_y = out_y + ky;
                        int in_x = out_x + kx;
                        
                        if (in_y < height && in_x < width) {
                            int in_idx = n * in_channels * height * width +
                                        c * height * width +
                                        in_y * width + in_x;
                            int weight_idx = out_c * in_channels * kernel_size * kernel_size +
                                            c * kernel_size * kernel_size +
                                            ky * kernel_size + kx;
                            sum += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            int out_idx = n * out_channels * out_height * out_width +
                         out_c * out_height * out_width +
                         out_y * out_width + out_x;
            
            sum += bias[out_c];
            sum *= multiplier[out_c];
            sum = sum > 0 ? sum : 0.01f * sum;
            sum = gelu_approx(sum);
            
            output[out_idx] = sum;
        }
    }
}

torch::Tensor fused_conv_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = height - kernel_size + 1;
    const int out_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   out_channels);
    
    fused_conv_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), multiplier.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width);
    
    return output;
}
"""

fused_conv_activation_cpp_source = (
    "torch::Tensor fused_conv_activation_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor multiplier);"
)

# Compile the inline CUDA code for fused operations
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
    """
    Optimized Model that fuses convolution, multiply, LeakyReLU, and GELU into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.fused_conv_activation = fused_conv_activation

    def forward(self, x):
        return self.fused_conv_activation.fused_conv_activation_cuda(
            x, self.conv.weight, self.conv.bias, self.multiplier
        )


batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]