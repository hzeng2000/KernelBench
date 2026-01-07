import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d + GELU + GroupNorm fusion
conv_transpose_gelu_groupnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 16

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void conv_transpose_gelu_groupnorm_kernel(
    const float* input, const float* weight, const float* bias,
    const float* gamma, const float* beta,
    float* output, float* mean, float* var,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_size, int stride, int num_groups) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    if (out_x >= out_width || out_y >= out_height || out_c >= out_channels) return;

    int group = out_c / (out_channels / num_groups);
    int group_start = group * (out_channels / num_groups);
    int group_size = out_channels / num_groups;

    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        for (int c = 0; c < in_channels; c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_x = (out_x + kx) / stride;
                    int in_y = (out_y + ky) / stride;
                    if (in_x < in_width && in_y < in_height) {
                        int in_idx = n * in_channels * in_height * in_width +
                                     c * in_height * in_width +
                                     in_y * in_width + in_x;
                        int w_idx = out_c * in_channels * kernel_size * kernel_size +
                                   c * kernel_size * kernel_size +
                                   ky * kernel_size + kx;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[out_c];
        }
        sum = gelu(sum);

        int out_idx = n * out_channels * out_height * out_width +
                     out_c * out_height * out_width +
                     out_y * out_width + out_x;
        output[out_idx] = sum;
    }
}

__global__ void compute_group_stats_kernel(
    const float* output, float* mean, float* var,
    int batch_size, int out_channels, int out_height, int out_width,
    int num_groups) {

    int group = blockIdx.x;
    int group_size = out_channels / num_groups;
    int group_start = group * group_size;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;

    for (int n = 0; n < batch_size; n++) {
        for (int c = group_start; c < group_start + group_size; c++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    int idx = n * out_channels * out_height * out_width +
                             c * out_height * out_width +
                             h * out_width + w;
                    float val = output[idx];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
        }
    }

    mean[group] = sum / count;
    var[group] = sum_sq / count - mean[group] * mean[group] + 1e-5f;
}

__global__ void group_norm_kernel(
    const float* output, const float* mean, const float* var,
    const float* gamma, const float* beta,
    float* normalized, int batch_size, int out_channels,
    int out_height, int out_width, int num_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * out_height * out_width;

    if (idx >= total_size) return;

    int n = idx / (out_channels * out_height * out_width);
    int c = (idx / (out_height * out_width)) % out_channels;
    int h = (idx / out_width) % out_height;
    int w = idx % out_width;

    int group = c / (out_channels / num_groups);
    float m = mean[group];
    float v = var[group];
    float gamma_val = gamma[c];
    float beta_val = beta[c];

    int out_idx = n * out_channels * out_height * out_width +
                 c * out_height * out_width +
                 h * out_width + w;
    float val = output[out_idx];
    normalized[out_idx] = gamma_val * ((val - m) / sqrtf(v)) + beta_val;
}

torch::Tensor conv_transpose_gelu_groupnorm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int stride, int num_groups) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    const auto out_height = in_height * stride;
    const auto out_width = in_width * stride;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    auto mean = torch::zeros({num_groups}, input.options());
    auto var = torch::zeros({num_groups}, input.options());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((out_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (out_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  out_channels);

    conv_transpose_gelu_groupnorm_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_size, stride, num_groups);

    compute_group_stats_kernel<<<num_groups, 1>>>(
        output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        batch_size, out_channels, out_height, out_width, num_groups);

    int total_size = batch_size * out_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    group_norm_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, out_channels,
        out_height, out_width, num_groups);

    return output;
}
"""

conv_transpose_gelu_groupnorm_cpp_source = """
torch::Tensor conv_transpose_gelu_groupnorm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor gamma, torch::Tensor beta,
    int stride, int num_groups);
"""

conv_transpose_gelu_groupnorm = load_inline(
    name="conv_transpose_gelu_groupnorm",
    cpp_sources=conv_transpose_gelu_groupnorm_cpp_source,
    cuda_sources=conv_transpose_gelu_groupnorm_source,
    functions=["conv_transpose_gelu_groupnorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.num_groups = num_groups
        self.conv_transpose_gelu_groupnorm = conv_transpose_gelu_groupnorm

    def forward(self, x):
        return self.conv_transpose_gelu_groupnorm.conv_transpose_gelu_groupnorm_cuda(
            x, self.weight, self.bias, self.gamma, self.beta, self.stride, self.num_groups)