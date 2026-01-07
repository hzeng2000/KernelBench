import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv2D + GELU + Global Average Pooling fusion
conv_gelu_gap_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 16

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void conv_gelu_gap_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_height, int out_width) {
    
    int out_c = blockIdx.z;
    int b = blockIdx.y;
    int out_h = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    int out_w = threadIdx.x;
    
    if (out_h < out_height && out_w < out_width) {
        float sum = 0.0f;
        
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_h = out_h + kh;
                    int in_w = out_w + kw;
                    
                    if (in_h < height && in_w < width) {
                        int input_idx = ((b * in_channels + in_c) * height + in_h) * width + in_w;
                        int weight_idx = ((out_c * in_channels + in_c) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[out_c];
        }
        
        // Apply GELU
        sum = gelu(sum);
        
        // Write intermediate result
        int inter_idx = ((b * out_channels + out_c) * out_height + out_h) * out_width + out_w;
        output[inter_idx] = sum;
    }
}

__global__ void global_avg_pool_kernel(
    const float* input, float* output,
    int batch_size, int channels, int height, int width) {
    
    int b = blockIdx.x;
    int c = blockIdx.y;
    
    float sum = 0.0f;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = ((b * channels + c) * height + h) * width + w;
            sum += input[idx];
        }
    }
    
    int out_idx = b * channels + c;
    output[out_idx] = sum / (height * width);
}

torch::Tensor conv_gelu_gap_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    
    auto intermediate = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    auto output = torch::zeros({batch_size, out_channels}, input.options());
    
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((out_height + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, out_channels);
    
    conv_gelu_gap_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        intermediate.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width);
    
    dim3 pool_grid(batch_size, out_channels);
    global_avg_pool_kernel<<<pool_grid, 1>>>(
        intermediate.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, out_channels, out_height, out_width);
    
    return output;
}
"""

conv_gelu_gap_cpp_source = "torch::Tensor conv_gelu_gap_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

conv_gelu_gap = load_inline(
    name="conv_gelu_gap",
    cpp_sources=conv_gelu_gap_cpp_source,
    cuda_sources=conv_gelu_gap_source,
    functions=["conv_gelu_gap_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_gelu_gap = conv_gelu_gap

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return self.conv_gelu_gap.conv_gelu_gap_cuda(x, weight, bias)