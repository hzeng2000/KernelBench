import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d + Swish fusion
conv_transpose_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_swish_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* swish_output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx < total_elements) {
        int tmp = idx;
        int b = tmp / (out_channels * out_depth * out_height * out_width);
        tmp %= (out_channels * out_depth * out_height * out_width);
        int c = tmp / (out_depth * out_height * out_width);
        tmp %= (out_depth * out_height * out_width);
        int d = tmp / (out_height * out_width);
        tmp %= (out_height * out_width);
        int h = tmp / out_width;
        int w = tmp % out_width;
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_d; kd++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int in_d = (d + pad_d - kd) / stride_d;
                        int in_h = (h + pad_h - kh) / stride_h;
                        int in_w = (w + pad_w - kw) / stride_w;
                        
                        if ((d + pad_d - kd) % stride_d == 0 && 
                            (h + pad_h - kh) % stride_h == 0 && 
                            (w + pad_w - kw) % stride_w == 0 &&
                            in_d >= 0 && in_d < in_depth &&
                            in_h >= 0 && in_h < in_height &&
                            in_w >= 0 && in_w < in_width) {
                            
                            int in_idx = b * in_channels * in_depth * in_height * in_width +
                                        ic * in_depth * in_height * in_width +
                                        in_d * in_height * in_width +
                                        in_h * in_width +
                                        in_w;
                            
                            int weight_idx = c * in_channels * kernel_d * kernel_h * kernel_w +
                                            ic * kernel_d * kernel_h * kernel_w +
                                            kd * kernel_h * kernel_w +
                                            kh * kernel_w +
                                            kw;
                            
                            sum += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[idx] = sum;
        
        // Swish activation: x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + expf(-sum));
        swish_output[idx] = sum * sigmoid;
    }
}

torch::Tensor conv_transpose_swish_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_depth = input.size(2);
    auto in_height = input.size(3);
    auto in_width = input.size(4);
    auto out_channels = weight.size(0);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d;
    auto out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h;
    auto out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    auto swish_output = torch::zeros_like(output);
    
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_transpose_swish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), swish_output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w);
    
    return swish_output;
}
"""

conv_transpose_swish_cpp_source = (
    "torch::Tensor conv_transpose_swish_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int stride_d, int stride_h, int stride_w,"
    "int pad_d, int pad_h, int pad_w);"
)

# Custom CUDA kernel for GroupNorm + HardSwish fusion
groupnorm_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void groupnorm_hardswish_kernel(
    const float* input, float* output,
    int batch_size, int groups, int channels,
    int depth, int height, int width,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_per_group = (batch_size * channels * depth * height * width) / groups;
    int total_elements = batch_size * channels * depth * height * width;
    
    if (idx < total_elements) {
        int group_idx = idx / elements_per_group;
        int group_start = group_idx * elements_per_group;
        int group_end = (group_idx + 1) * elements_per_group;
        
        // Compute mean for this group
        float sum = 0.0f;
        for (int i = group_start; i < group_end; i++) {
            sum += input[i];
        }
        float mean = sum / elements_per_group;
        
        // Compute variance for this group
        float var_sum = 0.0f;
        for (int i = group_start; i < group_end; i++) {
            float diff = input[i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / elements_per_group;
        float inv_std = rsqrtf(var + eps);
        
        // Normalize and apply HardSwish
        float x = input[idx];
        float normalized = (x - mean) * inv_std;
        
        // HardSwish: x * relu6(x + 3) / 6
        float x_plus_3 = normalized + 3.0f;
        float relu6 = fminf(fmaxf(x_plus_3, 0.0f), 6.0f);
        output[idx] = normalized * relu6 / 6.0f;
    }
}

torch::Tensor groupnorm_hardswish_cuda(torch::Tensor input, int groups, float eps) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    
    auto output = torch::zeros_like(input);
    
    int total_elements = batch_size * channels * depth * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    groupnorm_hardswish_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, groups, channels, depth, height, width, eps);
    
    return output;
}
"""

groupnorm_hardswish_cpp_source = (
    "torch::Tensor groupnorm_hardswish_cuda(torch::Tensor input, int groups, float eps);"
)

# Compile the inline CUDA code
conv_transpose_swish = load_inline(
    name="conv_transpose_swish",
    cpp_sources=conv_transpose_swish_cpp_source,
    cuda_sources=conv_transpose_swish_source,
    functions=["conv_transpose_swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

groupnorm_hardswish = load_inline(
    name="groupnorm_hardswish",
    cpp_sources=groupnorm_hardswish_cpp_source,
    cuda_sources=groupnorm_hardswish_source,
    functions=["groupnorm_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.groups = groups
        self.eps = eps
        self.stride = stride
        self.padding = padding
        self.conv_transpose_swish = conv_transpose_swish
        self.groupnorm_hardswish = groupnorm_hardswish

    def forward(self, x):
        x = self.conv_transpose_swish.conv_transpose_swish_cuda(
            x, self.weight, self.bias, self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding)
        x = self.groupnorm_hardswish.groupnorm_hardswish_cuda(x, self.groups, self.eps)
        return x