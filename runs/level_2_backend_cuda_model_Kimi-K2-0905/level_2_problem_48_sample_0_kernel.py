import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused 3D conv + scale + tanh + scale + sigmoid
fused_conv3d_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 8

__global__ void fused_conv3d_activation_kernel(
    const float* input, const float* weight, const float* bias,
    const float* scaling_factor, const float* bias_scale,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int pad, int stride) {
    
    int out_d = blockIdx.z * blockDim.z + threadIdx.z;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    
    int out_depth = (depth + 2 * pad - kernel_size) / stride + 1;
    int out_height = (height + 2 * pad - kernel_size) / stride + 1;
    int out_width = (width + 2 * pad - kernel_size) / stride + 1;
    
    if (out_d >= out_depth || out_h >= out_height || out_w >= out_width) return;
    
    int batch = blockIdx.w;
    int out_c = blockIdx.w / batch_size;
    batch = blockIdx.w % batch_size;
    
    float sum = 0.0f;
    
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int k_d = 0; k_d < kernel_size; ++k_d) {
            for (int k_h = 0; k_h < kernel_size; ++k_h) {
                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                    int in_d = out_d * stride - pad + k_d;
                    int in_h = out_h * stride - pad + k_h;
                    int in_w = out_w * stride - pad + k_w;
                    
                    if (in_d >= 0 && in_d < depth && in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        int input_idx = ((batch * in_channels + in_c) * depth + in_d) * height * width + in_h * width + in_w;
                        int weight_idx = ((out_c * in_channels + in_c) * kernel_size + k_d) * kernel_size * kernel_size + k_h * kernel_size + k_w;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[out_c];
    }
    
    // Apply scaling factor
    sum *= scaling_factor[out_c];
    
    // Apply tanh
    sum = tanhf(sum);
    
    // Apply bias scale
    sum *= bias_scale[out_c];
    
    // Apply sigmoid
    sum = 1.0f / (1.0f + expf(-sum));
    
    int output_idx = ((batch * out_channels + out_c) * out_depth + out_d) * out_height * out_width + out_h * out_width + out_w;
    output[output_idx] = sum;
}

torch::Tensor fused_conv3d_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor scaling_factor, torch::Tensor bias_scale) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int pad = 0;
    const int stride = 1;
    
    const int out_depth = (depth + 2 * pad - kernel_size) / stride + 1;
    const int out_height = (height + 2 * pad - kernel_size) / stride + 1;
    const int out_width = (width + 2 * pad - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 grid_size((out_width + TILE_SIZE - 1) / TILE_SIZE,
                   (out_height + TILE_SIZE - 1) / TILE_SIZE,
                   (out_depth + TILE_SIZE - 1) / TILE_SIZE);
    dim3 grid_dim(grid_size.x, grid_size.y, grid_size.z * batch_size * out_channels);
    
    fused_conv3d_activation_kernel<<<grid_dim, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        scaling_factor.data_ptr<float>(), bias_scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, pad, stride);
    
    return output;
}
"""

fused_conv3d_activation_cpp_source = (
    "torch::Tensor fused_conv3d_activation_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "torch::Tensor scaling_factor, torch::Tensor bias_scale);"
)

# Compile the inline CUDA code
fused_conv3d_activation = load_inline(
    name="fused_conv3d_activation",
    cpp_sources=fused_conv3d_activation_cpp_source,
    cuda_sources=fused_conv3d_activation_source,
    functions=["fused_conv3d_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a fused 3D convolution with activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_conv3d_activation

    def forward(self, x):
        return self.fused_op.fused_conv3d_activation_cuda(
            x, self.conv.weight, self.conv.bias,
            self.scaling_factor, self.bias
        )