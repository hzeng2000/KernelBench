import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for fused post-conv and custom group norm
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_post_conv_kernel(const float* input, const float* bias, const float* scale, float* output, int N, int C, int H, int W) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N && c < C && hw < H * W) {
        int idx = ((n * C + c) * H * W) + hw;
        float val = input[idx] + bias[c];
        val *= scale[c];
        output[idx] = 1.0f / (1.0f + expf(-val));
    }
}

torch::Tensor fused_post_conv_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto output = torch::empty_like(input);
    dim3 block(256);
    dim3 grid((H * W + 255) / 256, C, N);
    fused_post_conv_kernel<<<grid, block>>>(input.data_ptr<float>(), bias.data_ptr<float>(), scale.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);
    return output;
}

__global__ void compute_mean_var_kernel(const float* input, float* mean, float* var, int N, int C, int H, int W, int G) {
    int n = blockIdx.z;
    int g = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int channels_per_group = C / G;
    int num_elements = channels_per_group * H * W;
    int total_threads = gridDim.x * blockDim.x;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = tid; i < num_elements; i += total_threads) {
        int c_in_group = i / (H * W);
        int c = g * channels_per_group + c_in_group;
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((n * C + c) * H * W) + h * W + w;
        float val = input[idx];
        local_sum += val;
        local_sumsq += val * val;
    }
    __shared__ float shared_sum[256];
    __shared__ float shared_sumsq[256];
    shared_sum[threadIdx.x] = local_sum;
    shared_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            shared_sumsq[threadIdx.x] += shared_sumsq[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&mean[n * G + g], shared_sum[0]);
        atomicAdd(&var[n * G + g], shared_sumsq[0]);
    }
}

__global__ void normalize_kernel(const float* input, const float* mean, const float* var, const float* weight, const float* bias, float* output, int N, int C, int H, int W, int G, float eps) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int g = c / (C / G);
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N && c < C && hw < H * W) {
        int idx = ((n * C + c) * H * W) + hw;
        float m = mean[n * G + g];
        float v = var[n * G + g];
        float val = input[idx];
        val = (val - m) / sqrtf(v + eps);
        val = val * weight[c] + bias[c];
        output[idx] = val;
    }
}

torch::Tensor custom_group_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int G, float eps) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    auto output = torch::empty_like(input);
    auto mean = torch::zeros({N, G}, input.options());
    auto var = torch::zeros({N, G}, input.options());
    int channels_per_group = C / G;
    int num_elements = channels_per_group * H * W;
    int block_size = 256;
    int num_blocks_per_group = (num_elements + block_size - 1) / block_size;
    dim3 grid_mean_var(num_blocks_per_group, G, N);
    compute_mean_var_kernel<<<grid_mean_var, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), N, C, H, W, G);
    mean = mean / num_elements;
    var = var / num_elements - mean * mean;
    dim3 grid_norm((H * W + 255) / 256, C, N);
    normalize_kernel<<<grid_norm, 256>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, G, eps);
    return output;
}
"""

cpp_source = (
    "torch::Tensor fused_post_conv_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);"
    "torch::Tensor custom_group_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int G, float eps);"
)

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_post_conv_cuda", "custom_group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, fused bias addition, scaling, and sigmoid, followed by custom group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_post_conv = custom_ops
        self.custom_group_norm = custom_ops
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_post_conv.fused_post_conv_cuda(x, self.bias, self.scale)
        x = self.custom_group_norm.custom_group_norm_cuda(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, 1e-5)
        return x