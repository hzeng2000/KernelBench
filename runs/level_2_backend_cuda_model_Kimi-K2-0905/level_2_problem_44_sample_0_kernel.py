import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for transposed convolution + scalar multiplication + double global average pooling
fused_conv_transpose_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int in_channels, int out_channels, int in_h, int in_w,
    int out_h, int out_w, int kernel_size, int stride, int padding, int output_padding, float multiplier) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (out_x < out_w && out_y < out_h && out_c < out_channels) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int in_c = 0; in_c < in_channels; ++in_c) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_y = (out_y + padding - ky * stride) / stride;
                        int in_x = (out_x + padding - kx * stride) / stride;
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            int input_idx = b * in_channels * in_h * in_w + in_c * in_h * in_w + in_y * in_w + in_x;
                            int weight_idx = out_c * in_channels * kernel_size * kernel_size + in_c * kernel_size * kernel_size + ky * kernel_size + kx;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            if (bias != nullptr) {
                sum += bias[out_c];
            }
            sum *= multiplier;
            
            int output_idx = b * out_channels * out_h * out_w + out_c * out_h * out_w + out_y * out_w + out_x;
            output[output_idx] = sum;
        }
    }
}

__global__ void global_avg_pool_kernel(const float* input, float* output, int batch_size, int channels, int h, int w) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b < batch_size && c < channels) {
        float sum = 0.0f;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = b * channels * h * w + c * h * w + y * w + x;
                sum += input[idx];
            }
        }
        float mean = sum / (h * w);
        output[b * channels + c] = mean;
    }
}

torch::Tensor fused_conv_transpose_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding, int output_padding, float multiplier) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_h = input.size(2);
    auto in_w = input.size(3);
    auto out_channels = weight.size(0);
    
    int out_h = (in_h - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_w = (in_w - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    auto pooled1 = torch::zeros({batch_size, out_channels}, input.options());
    auto pooled2 = torch::zeros({batch_size, out_channels}, input.options());
    
    dim3 blockSize(16, 16, 4);
    dim3 gridSize((out_w + blockSize.x - 1) / blockSize.x,
                  (out_h + blockSize.y - 1) / blockSize.y,
                  (out_channels + blockSize.z - 1) / blockSize.z);
    
    conv_transpose_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w, kernel_size, stride, padding, output_padding, multiplier);
    
    dim3 poolBlockSize(16, 16);
    dim3 poolGridSize((batch_size + poolBlockSize.x - 1) / poolBlockSize.x,
                      (out_channels + poolBlockSize.y - 1) / poolBlockSize.y);
    
    global_avg_pool_kernel<<<poolGridSize, poolBlockSize>>>(
        output.data_ptr<float>(), pooled1.data_ptr<float>(), batch_size, out_channels, out_h, out_w);
    
    global_avg_pool_kernel<<<poolGridSize, poolBlockSize>>>(
        pooled1.data_ptr<float>(), pooled2.data_ptr<float>(), batch_size, out_channels, 1, 1);
    
    return pooled2.view({batch_size, out_channels, 1, 1});
}
"""

fused_conv_transpose_pool_cpp_source = (
    "torch::Tensor fused_conv_transpose_pool_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias,"
    "int kernel_size, int stride, int padding, int output_padding, float multiplier);"
)

fused_conv_transpose_pool = load_inline(
    name="fused_conv_transpose_pool",
    cpp_sources=fused_conv_transpose_pool_cpp_source,
    cuda_sources=fused_conv_transpose_pool_source,
    functions=["fused_conv_transpose_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.fused_op = fused_conv_transpose_pool

    def forward(self, x):
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return self.fused_op.fused_conv_transpose_pool_cuda(
            x, weight, bias, self.conv_transpose.kernel_size[0], self.conv_transpose.stride[0],
            self.conv_transpose.padding[0], self.conv_transpose.output_padding[0], self.multiplier
        )