import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused ConvTranspose2d + Mish + Add + Hardtanh + Scale
fused_conv_transpose_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__device__ float mish_activation(float x) {
    return x * tanhf(fmaxf(0.0f, x));
}

__device__ float hardtanh_activation(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

__global__ void conv_transpose_mish_add_hardtanh_scale_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_size, int stride, int padding, int output_padding,
    float add_value, float scale, float min_val, float max_val) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;
    
    if (out_x < out_width && out_y < out_height && out_c < out_channels) {
        float sum = 0.0f;
        
        for (int n = 0; n < batch_size; n++) {
            for (int in_c = 0; in_c < in_channels; in_c++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_x = (out_x + padding - kx) / stride;
                        int in_y = (out_y + padding - ky) / stride;
                        
                        if ((out_x + padding - kx) % stride == 0 && 
                            (out_y + padding - ky) % stride == 0 &&
                            in_x >= 0 && in_x < in_width && 
                            in_y >= 0 && in_y < in_height) {
                            
                            int input_idx = n * in_channels * in_height * in_width + 
                                          in_c * in_height * in_width + 
                                          in_y * in_width + in_x;
                            int weight_idx = out_c * in_channels * kernel_size * kernel_size + 
                                           in_c * kernel_size * kernel_size + 
                                           ky * kernel_size + kx;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            if (bias != nullptr) {
                sum += bias[out_c];
            }
            
            // Apply Mish activation
            sum = mish_activation(sum);
            
            // Add value
            sum += add_value;
            
            // Apply Hardtanh activation
            sum = hardtanh_activation(sum, min_val, max_val);
            
            // Scale
            sum *= scale;
            
            int output_idx = n * out_channels * out_height * out_width + 
                           out_c * out_height * out_width + 
                           out_y * out_width + out_x;
            output[output_idx] = sum;
        }
    }
}

torch::Tensor fused_conv_transpose_mish_add_hardtanh_scale_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int output_padding, float add_value, float scale) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               input.options());
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((out_width + TILE_SIZE - 1) / TILE_SIZE,
                   (out_height + TILE_SIZE - 1) / TILE_SIZE,
                   out_channels);
    
    conv_transpose_mish_add_hardtanh_scale_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, output_padding,
        add_value, scale, -1.0f, 1.0f);
    
    return output;
}
"""

fused_conv_transpose_activation_cpp_source = (
    "torch::Tensor fused_conv_transpose_mish_add_hardtanh_scale_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int stride, int padding, int output_padding, float add_value, float scale);"
)

# Compile the inline CUDA code
fused_conv_transpose_activation = load_inline(
    name="fused_conv_transpose_activation",
    cpp_sources=fused_conv_transpose_activation_cpp_source,
    cuda_sources=fused_conv_transpose_activation_source,
    functions=["fused_conv_transpose_mish_add_hardtanh_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.fused_op = fused_conv_transpose_activation

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_mish_add_hardtanh_scale_cuda(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.conv_transpose.stride[0], self.conv_transpose.padding[0], 
            self.conv_transpose.output_padding[0], self.add_value, self.scale)