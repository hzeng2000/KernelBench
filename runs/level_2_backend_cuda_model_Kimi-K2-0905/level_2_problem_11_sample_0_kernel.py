import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose + BatchNorm + Tanh fusion
conv_bn_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void conv_bn_tanh_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias, const float* bn_mean, const float* bn_var,
    float* output, 
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding, int out_height, int out_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_size) {
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % out_channels;
        int b = idx / (out_width * out_height * out_channels);
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_h = h * stride - padding + kh;
                    int in_w = w * stride - padding + kw;
                    
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        int in_idx = b * in_channels * height * width + ic * height * width + in_h * width + in_w;
                        int weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias) {
            sum += bias[c];
        }
        
        // BatchNorm
        float bn_scale = bn_weight[c] / sqrtf(bn_var[c] + 1e-5f);
        float bn_shift = bn_bias[c] - bn_weight[c] * bn_mean[c] / sqrtf(bn_var[c] + 1e-5f);
        sum = sum * bn_scale + bn_shift;
        
        // Tanh
        sum = tanhf(sum);
        
        output[idx] = sum;
    }
}

torch::Tensor conv_bn_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    int kernel_size, int stride, int padding) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    int out_height = (height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (width - 1) * stride - 2 * padding + kernel_size;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    int total_size = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    conv_bn_tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(), bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, out_height, out_width);
    
    return output;
}
"""

conv_bn_tanh_cpp_source = (
    "torch::Tensor conv_bn_tanh_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,"
    "int kernel_size, int stride, int padding);"
)

conv_bn_tanh = load_inline(
    name="conv_bn_tanh",
    cpp_sources=conv_bn_tanh_cpp_source,
    cuda_sources=conv_bn_tanh_source,
    functions=["conv_bn_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for MaxPool + GroupNorm fusion
maxpool_groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void maxpool_groupnorm_kernel(
    const float* input, const float* gn_weight, const float* gn_bias,
    float* output, 
    int batch_size, int channels, int height, int width, int num_groups) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = height / 2;
    int out_width = width / 2;
    int total_size = batch_size * channels * out_height * out_width;
    
    if (idx < total_size) {
        int w = idx % out_width;
        int h = (idx / out_width) % out_height;
        int c = (idx / (out_width * out_height)) % channels;
        int b = idx / (out_width * out_height * channels);
        
        // MaxPool
        float max_val = -INFINITY;
        for (int kh = 0; kh < 2; kh++) {
            for (int kw = 0; kw < 2; kw++) {
                int in_h = h * 2 + kh;
                int in_w = w * 2 + kw;
                int in_idx = b * channels * height * width + c * height * width + in_h * width + in_w;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        
        // GroupNorm
        int group_size = channels / num_groups;
        int group_idx = c / group_size;
        
        // Compute mean and var for the group
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int gc = group_idx * group_size; gc < (group_idx + 1) * group_size; gc++) {
            for (int gh = 0; gh < out_height; gh++) {
                for (int gw = 0; gw < out_width; gw++) {
                    int g_idx = b * channels * out_height * out_width + gc * out_height * out_width + gh * out_width + gw;
                    sum += input[g_idx];
                    sum_sq += input[g_idx] * input[g_idx];
                }
            }
        }
        
        float mean = sum / (group_size * out_height * out_width);
        float var = sum_sq / (group_size * out_height * out_width) - mean * mean;
        
        float gn_scale = gn_weight[c] / sqrtf(var + 1e-5f);
        float gn_shift = gn_bias[c];
        
        max_val = (max_val - mean) * gn_scale + gn_shift;
        
        output[idx] = max_val;
    }
}

torch::Tensor maxpool_groupnorm_cuda(
    torch::Tensor input, torch::Tensor gn_weight, torch::Tensor gn_bias, int num_groups) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    int out_height = height / 2;
    int out_width = width / 2;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    int total_size = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    maxpool_groupnorm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), gn_weight.data_ptr<float>(), gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width, num_groups);
    
    return output;
}
"""

maxpool_groupnorm_cpp_source = (
    "torch::Tensor maxpool_groupnorm_cuda("
    "torch::Tensor input, torch::Tensor gn_weight, torch::Tensor gn_bias, int num_groups);"
)

maxpool_groupnorm = load_inline(
    name="maxpool_groupnorm",
    cpp_sources=maxpool_groupnorm_cpp_source,
    cuda_sources=maxpool_groupnorm_source,
    functions=["maxpool_groupnorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        self.conv_bn_tanh = conv_bn_tanh
        self.maxpool_groupnorm = maxpool_groupnorm

    def forward(self, x):
        x = self.conv_bn_tanh.conv_bn_tanh_cuda(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.batch_norm.weight, self.batch_norm.bias, self.batch_norm.running_mean, self.batch_norm.running_var,
            self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0], self.conv_transpose.padding[0]
        )
        x = self.maxpool_groupnorm.maxpool_groupnorm_cuda(
            x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups
        )
        return x