import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused conv + min + tanh + tanh
fused_conv_min_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void fused_conv_min_tanh_tanh_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size, int out_height, int out_width) {
    
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int b = blockIdx.z;
    
    if (out_x < out_width && out_y < out_height && b < batch_size) {
        float min_val = 1e20f;
        
        for (int oc = 0; oc < out_channels; oc++) {
            float sum = 0.0f;
            
            for (int ic = 0; ic < in_channels; ic++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_y = out_y + ky;
                        int in_x = out_x + kx;
                        
                        if (in_y < height && in_x < width) {
                            int input_idx = ((b * in_channels + ic) * height + in_y) * width + in_x;
                            int weight_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            if (bias != nullptr) {
                sum += bias[oc];
            }
            
            if (sum < min_val) {
                min_val = sum;
            }
        }
        
        // Apply tanh twice
        min_val = tanhf(min_val);
        min_val = tanhf(min_val);
        
        int out_idx = (b * out_height + out_y) * out_width + out_x;
        output[out_idx] = min_val;
    }
}

torch::Tensor fused_conv_min_tanh_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int out_height = height - kernel_size + 1;
    int out_width = width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, 1, out_height, out_width}, input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((out_width + TILE_SIZE - 1) / TILE_SIZE,
                   (out_height + TILE_SIZE - 1) / TILE_SIZE,
                   batch_size);
    
    fused_conv_min_tanh_tanh_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        height, width, kernel_size, out_height, out_width);
    
    return output;
}
"""

fused_conv_min_tanh_cpp_source = """
torch::Tensor fused_conv_min_tanh_tanh_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_conv_min_tanh = load_inline(
    name="fused_conv_min_tanh",
    cpp_sources=fused_conv_min_tanh_cpp_source,
    cuda_sources=fused_conv_min_tanh_source,
    functions=["fused_conv_min_tanh_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_op = fused_conv_min_tanh

    def forward(self, x):
        weight = self.conv.weight
        bias = self.conv.bias
        return self.fused_op.fused_conv_min_tanh_tanh_cuda(x, weight, bias)