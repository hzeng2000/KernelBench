import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv2D + BatchNorm + Scale fusion
conv_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_bn_scale_kernel(
    const float* input, const float* weight, const float* bias,
    const float* bn_weight, const float* bn_bias, const float* bn_mean, const float* bn_var,
    float* output, float scaling_factor,
    int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    
    int out_h = height - kernel_size + 1;
    int out_w = width - kernel_size + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;
    
    if (idx < total_elements) {
        int n = idx / (out_channels * out_h * out_w);
        int c = (idx / (out_h * out_w)) % out_channels;
        int h = (idx / out_w) % out_h;
        int w = idx % out_w;
        
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_idx = n * in_channels * height * width + ic * height * width + (h + kh) * width + (w + kw);
                    int weight_idx = c * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
        
        // Add bias
        sum += bias[c];
        
        // BatchNorm
        float bn_scale = bn_weight[c] / sqrtf(bn_var[c] + 1e-5f);
        float bn_shift = bn_bias[c] - bn_weight[c] * bn_mean[c] / sqrtf(bn_var[c] + 1e-5f);
        sum = bn_scale * sum + bn_shift;
        
        // Scale
        sum *= scaling_factor;
        
        int out_idx = n * out_channels * out_h * out_w + c * out_h * out_w + h * out_w + w;
        output[out_idx] = sum;
    }
}

torch::Tensor conv_bn_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var,
    float scaling_factor) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_h = height - kernel_size + 1;
    int out_w = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    int total_elements = batch_size * out_channels * out_h * out_w;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_bn_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(), bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        output.data_ptr<float>(), scaling_factor,
        batch_size, in_channels, out_channels, height, width, kernel_size);
    
    return output;
}
"""

conv_bn_scale_cpp_source = "torch::Tensor conv_bn_scale_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, float scaling_factor);"

conv_bn_scale = load_inline(
    name="conv_bn_scale",
    cpp_sources=conv_bn_scale_cpp_source,
    cuda_sources=conv_bn_scale_source,
    functions=["conv_bn_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.conv_bn_scale = conv_bn_scale

    def forward(self, x):
        return self.conv_bn_scale.conv_bn_scale_cuda(
            x, self.conv.weight, self.conv.bias,
            self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var,
            self.scaling_factor
        )