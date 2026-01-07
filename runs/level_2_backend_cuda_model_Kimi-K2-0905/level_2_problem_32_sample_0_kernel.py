import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv2d + scale + min
fused_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void fused_conv_scale_min_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, float scale_factor) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    int b = blockIdx.w;
    
    if (out_x < width && out_y < height && out_c < out_channels && b < batch_size) {
        float sum = 0.0f;
        int pad = kernel_size / 2;
        
        for (int in_c = 0; in_c < in_channels; in_c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = out_y - pad + ky;
                    int in_x = out_x - pad + kx;
                    
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int in_idx = b * in_channels * height * width + 
                                    in_c * height * width + in_y * width + in_x;
                        int weight_idx = out_c * in_channels * kernel_size * kernel_size +
                                        in_c * kernel_size * kernel_size + ky * kernel_size + kx;
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[out_c];
        }
        
        sum *= scale_factor;
        
        int out_idx = b * out_channels * height * width + out_c * height * width + out_y * width + out_x;
        output[out_idx] = sum;
    }
}

__global__ void channel_min_kernel(
    const float* input, float* output, int batch_size, int channels,
    int height, int width) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    
    if (x < width && y < height && b < batch_size) {
        float min_val = FLT_MAX;
        
        for (int c = 0; c < channels; c++) {
            int idx = b * channels * height * width + c * height * width + y * width + x;
            min_val = fminf(min_val, input[idx]);
        }
        
        int out_idx = b * height * width + y * width + x;
        output[out_idx] = min_val;
    }
}

torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float scale_factor) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto conv_output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (height + TILE_WIDTH - 1) / TILE_WIDTH,
                 out_channels,
                 batch_size);
    
    fused_conv_scale_min_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        conv_output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, scale_factor);
    
    auto min_output = torch::zeros({batch_size, 1, height, width}, input.options());
    
    dim3 minBlockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 minGridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                    (height + TILE_WIDTH - 1) / TILE_WIDTH,
                    batch_size);
    
    channel_min_kernel<<<minGridDim, minBlockDim>>>(
        conv_output.data_ptr<float>(), min_output.data_ptr<float>(),
        batch_size, out_channels, height, width);
    
    return min_output;
}
"""

fused_conv_scale_min_cpp_source = (
    "torch::Tensor fused_conv_scale_min_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scale_factor);"
)

# Compile the inline CUDA code
fused_conv_scale_min = load_inline(
    name="fused_conv_scale_min",
    cpp_sources=fused_conv_scale_min_cpp_source,
    cuda_sources=fused_conv_scale_min_source,
    functions=["fused_conv_scale_min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.fused_op = fused_conv_scale_min

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias if self.conv.bias is not None else torch.empty(0).cuda()
        return self.fused_op.fused_conv_scale_min_cuda(x, weight, bias, self.scale_factor)