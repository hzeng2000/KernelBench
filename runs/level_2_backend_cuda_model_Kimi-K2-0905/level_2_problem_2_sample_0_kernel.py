import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused ConvTranspose2d + bias + clamp + scale + clamp + divide
conv_transpose_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_fused_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_size, int stride, int padding, int output_padding,
    float scaling_factor) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (out_x < out_width && out_y < out_height && out_c < out_channels) {
        float sum = 0.0f;
        
        for (int b = 0; b < batch_size; b++) {
            for (int ic = 0; ic < in_channels; ic++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_y = (out_y + padding - ky) / stride;
                        int in_x = (out_x + padding - kx) / stride;
                        
                        if ((out_y + padding - ky) % stride == 0 && 
                            (out_x + padding - kx) % stride == 0 &&
                            in_y >= 0 && in_y < in_height &&
                            in_x >= 0 && in_x < in_width) {
                            
                            int input_idx = b * in_channels * in_height * in_width +
                                          ic * in_height * in_width +
                                          in_y * in_width + in_x;
                            int weight_idx = out_c * in_channels * kernel_size * kernel_size +
                                           ic * kernel_size * kernel_size +
                                           ky * kernel_size + kx;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            int output_idx = b * out_channels * out_height * out_width +
                           out_c * out_height * out_width +
                           out_y * out_width + out_x;
            
            sum += bias[out_c];
            sum = fminf(fmaxf(sum, 0.0f), 1.0f);
            sum = sum * scaling_factor;
            sum = fminf(fmaxf(sum, 0.0f), 1.0f);
            sum = sum / scaling_factor;
            
            output[output_idx] = sum;
        }
    }
}

torch::Tensor conv_transpose_fused_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding, int output_padding,
    float scaling_factor) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_channels = weight.size(0);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 block_size(16, 16, 4);
    dim3 grid_size((out_width + block_size.x - 1) / block_size.x,
                   (out_height + block_size.y - 1) / block_size.y,
                   (out_channels + block_size.z - 1) / block_size.z);
    
    conv_transpose_fused_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, padding, output_padding, scaling_factor);
    
    return output;
}
"""

conv_transpose_fused_cpp_source = (
    "torch::Tensor conv_transpose_fused_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int kernel_size, int stride, int padding, int output_padding,"
    "float scaling_factor);"
)

conv_transpose_fused = load_inline(
    name="conv_transpose_fused",
    cpp_sources=conv_transpose_fused_cpp_source,
    cuda_sources=conv_transpose_fused_source,
    functions=["conv_transpose_fused_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        self.conv_transpose_fused = conv_transpose_fused
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        return self.conv_transpose_fused.conv_transpose_fused_cuda(
            x, self.conv_transpose.weight, self.bias,
            self.kernel_size, self.stride, self.padding, self.output_padding,
            self.scaling_factor
        )