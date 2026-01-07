import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + subtract + hardswish + maxpool + mish
fused_conv_block_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16

__device__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__device__ float mish(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_conv_block_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size, float subtract_value,
    int pool_kernel_size, int out_height, int out_width) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.w;
    
    if (out_x >= out_width || out_y >= out_height || out_c >= out_channels || b >= batch_size) return;
    
    float sum = 0.0f;
    int pad = kernel_size / 2;
    
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = out_x * pool_kernel_size - pad + kx;
                int in_y = out_y * pool_kernel_size - pad + ky;
                
                if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
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
    
    sum -= subtract_value;
    sum = hardswish(sum);
    
    int output_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
    output[output_idx] = mish(sum);
}
"""

fused_conv_block_cpp_source = """
torch::Tensor fused_conv_block_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float subtract_value, int pool_kernel_size);
"""

# Compile the inline CUDA code
fused_conv_block = load_inline(
    name="fused_conv_block",
    cpp_sources=fused_conv_block_cpp_source,
    cuda_sources=fused_conv_block_source,
    functions=["fused_conv_block_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        self.fused_conv_block = fused_conv_block
        
        # Initialize conv weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        in_height = x.size(2)
        in_width = x.size(3)
        out_channels = self.conv.weight.size(0)
        
        out_height = in_height // self.pool_kernel_size
        out_width = in_width // self.pool_kernel_size
        
        # Use fused kernel
        output = torch.zeros(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
        
        # Launch fused kernel
        self.fused_conv_block.fused_conv_block_cuda(
            x, self.conv.weight, self.conv.bias,
            self.subtract_value, self.pool_kernel_size
        )
        
        return output

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]