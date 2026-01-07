import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused operations: min + sum + gelu + add
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_min_sum_gelu_add_kernel(
    const float* input, float* output, const float* bias,
    int batch_size, int channels, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * 1 * 1 * width;
    
    if (idx < total_elements) {
        int b = idx / width;
        int w = idx % width;
        
        // Min operation along channel dimension
        float min_val = input[b * channels * height * width + w];
        for (int c = 1; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                float val = input[b * channels * height * width + c * height * width + h * width + w];
                if (val < min_val) min_val = val;
            }
        }
        
        // Sum operation along height dimension (already reduced to 1)
        float sum_val = min_val * height;
        
        // GELU activation
        float gelu_val = 0.5f * sum_val * (1.0f + tanhf(0.7978845608f * (sum_val + 0.044715f * sum_val * sum_val * sum_val)));
        
        // Add bias
        output[idx] = gelu_val + bias[0];
    }
}

torch::Tensor fused_min_sum_gelu_add_cuda(torch::Tensor input, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto output = torch::zeros({batch_size, 1, 1, width}, input.options());
    
    int total_elements = batch_size * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_min_sum_gelu_add_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, channels, height, width);
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_min_sum_gelu_add_cuda(torch::Tensor input, torch::Tensor bias);"

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_min_sum_gelu_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Custom CUDA kernel for ConvTranspose2d
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_size, int stride, int padding, int output_padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx < total_elements) {
        int b = idx / (out_channels * out_height * out_width);
        int c = (idx / (out_height * out_width)) % out_channels;
        int h = (idx / out_width) % out_height;
        int w = idx % out_width;
        
        float sum = 0.0f;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_h = (h + padding - kh) / stride;
                    int in_w = (w + padding - kw) / stride;
                    
                    if ((h + padding - kh) % stride == 0 && (w + padding - kw) % stride == 0 &&
                        in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                        
                        float input_val = input[b * in_channels * in_height * in_width + 
                                              ic * in_height * in_width + 
                                              in_h * in_width + in_w];
                        float weight_val = weight[c * in_channels * kernel_size * kernel_size + 
                                                ic * kernel_size * kernel_size + 
                                                kh * kernel_size + kw];
                        sum += input_val * weight_val;
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int stride = 2;
    int padding = 1;
    int output_padding = 1;
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv_transpose_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, output_padding);
    
    return output;
}
"""

conv_transpose_cpp_source = "torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight);"

conv_transpose_cuda = load_inline(
    name="conv_transpose_cuda",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops
        self.conv_transpose_cuda = conv_transpose_cuda

    def forward(self, x):
        x = self.conv_transpose_cuda.conv_transpose_cuda(x, self.conv_transpose.weight)
        x = self.fused_ops.fused_min_sum_gelu_add_cuda(x, self.bias)
        return x