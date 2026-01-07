import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv2D + Activation + BatchNorm fusion
conv_bn_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 16

__device__ float softplus(float x) {
    return logf(1.0f + expf(fminf(x, 20.0f)));
}

__device__ float tanh(float x) {
    float exp2x = expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

__global__ void conv_bn_fused_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias, const float* bn_mean, const float* bn_var,
    float* output, 
    int batch_size, int in_channels, int out_channels, 
    int height, int width, int kernel_size, int out_height, int out_width,
    float eps) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int b = blockIdx.z / out_channels;
    
    if (out_x < out_width && out_y < out_height && b < batch_size) {
        float sum = 0.0f;
        
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_x = out_x + kx;
                    int in_y = out_y + ky;
                    
                    if (in_x < width && in_y < height) {
                        int input_idx = ((b * in_channels + in_c) * height + in_y) * width + in_x;
                        int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        if (bias != nullptr) {
            sum += bias[out_c];
        }
        
        // Apply softplus + tanh * x activation
        float sp = softplus(sum);
        float tanh_sp = tanh(sp);
        float activated = tanh_sp * sum;
        
        // Apply batch normalization
        float bn_scale = bn_weight[out_c] / sqrtf(bn_var[out_c] + eps);
        float bn_shift = bn_bias[out_c] - bn_mean[out_c] * bn_scale;
        float normalized = activated * bn_scale + bn_shift;
        
        int output_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
        output[output_idx] = normalized;
    }
}

torch::Tensor conv_bn_fused_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    int kernel_size, float eps) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    
    const int out_height = height - kernel_size + 1;
    const int out_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_size * out_channels);
    
    conv_bn_fused_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(), 
        bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width, eps);
    
    return output;
}
"""

conv_bn_fused_cpp_source = (
    "torch::Tensor conv_bn_fused_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,"
    "int kernel_size, float eps);"
)

# Compile the inline CUDA code
conv_bn_fused = load_inline(
    name="conv_bn_fused",
    cpp_sources=conv_bn_fused_cpp_source,
    cuda_sources=conv_bn_fused_source,
    functions=["conv_bn_fused_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.conv_bn_fused = conv_bn_fused
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.conv_bn_fused.conv_bn_fused_cuda(
            x, self.conv.weight, self.conv.bias,
            self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var,
            self.kernel_size, self.eps
        )