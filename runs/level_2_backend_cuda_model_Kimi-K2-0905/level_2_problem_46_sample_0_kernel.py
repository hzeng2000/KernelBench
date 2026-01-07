import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + subtract1 + tanh + subtract2 + avgpool
fused_conv_tanh_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_conv_tanh_pool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size, int stride, int padding,
    float subtract1, float subtract2, int pool_kernel, int pool_stride, int pool_padding,
    int out_height, int out_width, int pooled_height, int pooled_width) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int b = blockIdx.z / out_channels;
    
    if (out_x < out_width && out_y < out_height && b < batch_size) {
        float sum = 0.0f;
        
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_y = out_y * stride - padding + ky;
                    int in_x = out_x * stride - padding + kx;
                    
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        int input_idx = ((b * in_channels + in_c) * in_height + in_y) * in_width + in_x;
                        int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[out_c];
        }
        
        // Subtract 1, tanh, subtract 2
        sum = tanhf(sum - subtract1) - subtract2;
        
        // Average pooling
        int pool_out_y = out_y / pool_stride;
        int pool_out_x = out_x / pool_stride;
        
        if (out_y % pool_stride == 0 && out_x % pool_stride == 0 && 
            pool_out_y < pooled_height && pool_out_x < pooled_width) {
            
            float pool_sum = 0.0f;
            int pool_count = 0;
            
            for (int py = 0; py < pool_kernel; ++py) {
                for (int px = 0; px < pool_kernel; ++px) {
                    int in_y = pool_out_y * pool_stride - pool_padding + py;
                    int in_x = pool_out_x * pool_stride - pool_padding + px;
                    
                    if (in_y >= 0 && in_y < out_height && in_x >= 0 && in_x < out_width) {
                        pool_sum += sum;
                        pool_count++;
                    }
                }
            }
            
            if (pool_count > 0) {
                int output_idx = ((b * out_channels + out_c) * pooled_height + pool_out_y) * pooled_width + pool_out_x;
                output[output_idx] = pool_sum / pool_count;
            }
        }
    }
}

torch::Tensor fused_conv_tanh_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding,
    float subtract1, float subtract2, int pool_kernel, int pool_stride, int pool_padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    int pooled_height = (out_height + 2 * pool_padding - pool_kernel) / pool_stride + 1;
    int pooled_width = (out_width + 2 * pool_padding - pool_kernel) / pool_stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, pooled_height, pooled_width}, input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((out_width + TILE_SIZE - 1) / TILE_SIZE,
                   (out_height + TILE_SIZE - 1) / TILE_SIZE,
                   batch_size * out_channels);
    
    fused_conv_tanh_pool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size, stride, padding,
        subtract1, subtract2, pool_kernel, pool_stride, pool_padding,
        out_height, out_width, pooled_height, pooled_width);
    
    return output;
}
"""

fused_conv_tanh_pool_cpp_source = (
    "torch::Tensor fused_conv_tanh_pool_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int kernel_size, int stride, int padding,"
    "float subtract1, float subtract2, int pool_kernel, int pool_stride, int pool_padding);"
)

# Compile the inline CUDA code
fused_conv_tanh_pool = load_inline(
    name="fused_conv_tanh_pool",
    cpp_sources=fused_conv_tanh_pool_cpp_source,
    cuda_sources=fused_conv_tanh_pool_source,
    functions=["fused_conv_tanh_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_op = fused_conv_tanh_pool

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else torch.empty(0)
        return self.fused_op.fused_conv_tanh_pool_cuda(
            x, weight, bias,
            self.conv.kernel_size[0], self.conv.stride[0], self.conv.padding[0],
            self.subtract1_value, self.subtract2_value,
            self.kernel_size_pool, self.kernel_size_pool, 0
        )