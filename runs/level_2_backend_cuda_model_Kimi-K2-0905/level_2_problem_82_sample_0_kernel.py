import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom fused CUDA kernel for conv+tanh+scale+bias+maxpool
fused_conv_tanh_scale_bias_maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void conv2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* conv_out, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_height, int out_width) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.w;

    if (out_x < out_width && out_y < out_height && out_c < out_channels && b < batch_size) {
        float sum = 0.0f;
        int pad = kernel_size / 2;
        
        for (int in_c = 0; in_c < in_channels; in_c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_x = out_x - pad + kx;
                    int in_y = out_y - pad + ky;
                    
                    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                        int input_idx = b * (in_channels * height * width) + 
                                       in_c * (height * width) + 
                                       in_y * width + in_x;
                        int weight_idx = out_c * (in_channels * kernel_size * kernel_size) +
                                        in_c * (kernel_size * kernel_size) +
                                        ky * kernel_size + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        int out_idx = b * (out_channels * out_height * out_width) +
                     out_c * (out_height * out_width) +
                     out_y * out_width + out_x;
        conv_out[out_idx] = sum + (bias ? bias[out_c] : 0.0f);
    }
}

__global__ void tanh_scale_bias_maxpool_kernel(
    const float* conv_out, const float* bias, float* final_out,
    int batch_size, int out_channels, int out_height, int out_width,
    int pool_size, int pooled_height, int pooled_width,
    float scaling_factor) {
    
    int pooled_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pooled_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.w;

    if (pooled_x < pooled_width && pooled_y < pooled_height && out_c < out_channels && b < batch_size) {
        float max_val = -INFINITY;
        
        for (int py = 0; py < pool_size; py++) {
            for (int px = 0; px < pool_size; px++) {
                int out_x = pooled_x * pool_size + px;
                int out_y = pooled_y * pool_size + py;
                
                if (out_x < out_width && out_y < out_height) {
                    int conv_idx = b * (out_channels * out_height * out_width) +
                                  out_c * (out_height * out_width) +
                                  out_y * out_width + out_x;
                    
                    float tanh_val = tanhf(conv_out[conv_idx]);
                    float scaled_val = tanh_val * scaling_factor;
                    float biased_val = scaled_val + (bias ? bias[out_c] : 0.0f);
                    
                    if (biased_val > max_val) {
                        max_val = biased_val;
                    }
                }
            }
        }
        
        int final_idx = b * (out_channels * pooled_height * pooled_width) +
                       out_c * (pooled_height * pooled_width) +
                       pooled_y * pooled_width + pooled_x;
        final_out[final_idx] = max_val;
    }
}

torch::Tensor fused_conv_tanh_scale_bias_maxpool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, int pool_size) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    
    int pad = kernel_size / 2;
    int out_height = height;
    int out_width = width;
    int pooled_height = out_height / pool_size;
    int pooled_width = out_width / pool_size;
    
    auto conv_out = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                                input.options());
    auto final_out = torch::zeros({batch_size, out_channels, pooled_height, pooled_width}, 
                                 input.options());
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE, 8);
    dim3 gridSizeConv((out_width + blockSize.x - 1) / blockSize.x,
                      (out_height + blockSize.y - 1) / blockSize.y,
                      (out_channels + blockSize.z - 1) / blockSize.z,
                      batch_size);
    
    conv2d_kernel<<<gridSizeConv, blockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        conv_out.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width);
    
    dim3 gridSizePool((pooled_width + blockSize.x - 1) / blockSize.x,
                      (pooled_height + blockSize.y - 1) / blockSize.y,
                      (out_channels + blockSize.z - 1) / blockSize.z,
                      batch_size);
    
    tanh_scale_bias_maxpool_kernel<<<gridSizePool, blockSize>>>(
        conv_out.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        final_out.data_ptr<float>(), batch_size, out_channels, out_height, out_width,
        pool_size, pooled_height, pooled_width, scaling_factor);
    
    return final_out;
}
"""

fused_conv_tanh_scale_bias_maxpool_cpp_source = """
torch::Tensor fused_conv_tanh_scale_bias_maxpool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scaling_factor, int pool_size);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_conv_tanh_scale_bias_maxpool_cpp_source,
    cuda_sources=fused_conv_tanh_scale_bias_maxpool_source,
    functions=["fused_conv_tanh_scale_bias_maxpool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return self.fused_ops.fused_conv_tanh_scale_bias_maxpool_cuda(
            x, weight, bias, self.scaling_factor, self.pool_kernel_size)