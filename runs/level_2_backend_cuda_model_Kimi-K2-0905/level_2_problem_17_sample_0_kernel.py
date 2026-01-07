import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused conv2d + instance_norm + divide
fused_conv_norm_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void fused_conv_norm_div_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, float* running_mean, float* running_var,
    int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size,
    float divide_by) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    if (out_x < width && out_y < height && out_c < out_channels) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        int out_idx = ((blockIdx.z * batch_size + blockIdx.w) * height + out_y) * width + out_x;

        // Compute convolution output
        float conv_val = 0.0f;
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_x = out_x + kx;
                    int in_y = out_y + ky;
                    if (in_x < width && in_y < height) {
                        int in_idx = ((in_c * batch_size + blockIdx.w) * height + in_y) * width + in_x;
                        int w_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                        conv_val += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        if (bias) {
            conv_val += bias[out_c];
        }

        // Compute instance norm stats per channel
        __shared__ float shared_sum[256];
        __shared__ float shared_sum_sq[256];
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        shared_sum[tid] = conv_val;
        shared_sum_sq[tid] = conv_val * conv_val;
        __syncthreads();

        // Reduce within block
        for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
                shared_sum_sq[tid] += shared_sum_sq[tid + stride];
            }
            __syncthreads();
        }

        float mean = shared_sum[0] / (batch_size * height * width);
        float var = shared_sum_sq[0] / (batch_size * height * width) - mean * mean;
        float inv_std = rsqrtf(var + 1e-5f);

        // Apply instance norm and divide
        float norm_val = (conv_val - mean) * inv_std;
        output[out_idx] = norm_val / divide_by;
    }
}

torch::Tensor fused_conv_norm_div_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    float divide_by) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE,
              out_channels,
              batch_size);

    fused_conv_norm_div_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), nullptr, nullptr,
        batch_size, in_channels, out_channels, height, width, kernel_size,
        divide_by);

    return output;
}
"""

fused_conv_norm_div_cpp_source = (
    "torch::Tensor fused_conv_norm_div_cuda("
    "torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divide_by);"
)

# Compile the inline CUDA code
fused_conv_norm_div = load_inline(
    name="fused_conv_norm_div",
    cpp_sources=fused_conv_norm_div_cpp_source,
    cuda_sources=fused_conv_norm_div_source,
    functions=["fused_conv_norm_div_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model with fused CUDA kernel for conv2d + instance_norm + divide
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.fused_op = fused_conv_norm_div

    def forward(self, x):
        return self.fused_op.fused_conv_norm_div_cuda(
            x, self.conv.weight, self.conv.bias, self.divide_by
        )