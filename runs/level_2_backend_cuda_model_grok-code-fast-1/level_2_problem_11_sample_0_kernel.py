import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused batch normalization and tanh
batch_norm_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void batch_norm_tanh_kernel(const float* x, float* out, const float* gamma, const float* beta, const float* mean, const float* var, float eps, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * H * W) {
        int c = (idx / (H * W)) % C;
        float val = x[idx];
        val = gamma[c] * (val - mean[c]) / sqrtf(var[c] + eps) + beta[c];
        out[idx] = tanhf(val);
    }
}

torch::Tensor batch_norm_tanh_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor var, float eps) {
    auto out = torch::empty_like(x);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int size = N * C * H * W;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    batch_norm_tanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), eps, N, C, H, W);
    return out;
}
"""

batch_norm_tanh_cpp_source = (
    "torch::Tensor batch_norm_tanh_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor mean, torch::Tensor var, float eps);"
)

# Compile the inline CUDA code for fused batch normalization and tanh
batch_norm_tanh = load_inline(
    name="batch_norm_tanh",
    cpp_sources=batch_norm_tanh_cpp_source,
    cuda_sources=batch_norm_tanh_source,
    functions=["batch_norm_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for max pooling
max_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool_kernel(const float* x, float* out, int N, int C, int H, int W, int out_H, int out_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_H * out_W;
    if (idx < total) {
        int n = idx / (C * out_H * out_W);
        int c = (idx / (out_H * out_W)) % C;
        int oh = (idx / out_W) % out_H;
        int ow = idx % out_W;
        int ih = oh * 2;
        int iw = ow * 2;
        float max_val = -INFINITY;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int h = ih + i;
                int w = iw + j;
                if (h < H && w < W) {
                    int in_idx = ((n * C + c) * H + h) * W + w;
                    max_val = fmaxf(max_val, x[in_idx]);
                }
            }
        }
        int out_idx = ((n * C + c) * out_H + oh) * out_W + ow;
        out[out_idx] = max_val;
    }
}

torch::Tensor max_pool_cuda(torch::Tensor x) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int out_H = H / 2;
    int out_W = W / 2;
    auto out = torch::empty({N, C, out_H, out_W}, x.options());
    int size = N * C * out_H * out_W;
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    max_pool_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, out_H, out_W);
    return out;
}
"""

max_pool_cpp_source = (
    "torch::Tensor max_pool_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for max pooling
max_pool = load_inline(
    name="max_pool",
    cpp_sources=max_pool_cpp_source,
    cuda_sources=max_pool_source,
    functions=["max_pool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, fused batch normalization and tanh, custom max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.batch_norm_tanh = batch_norm_tanh
        self.max_pool = max_pool
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm_tanh.batch_norm_tanh_cuda(x, self.batch_norm.weight, self.batch_norm.bias, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.eps)
        x = self.max_pool.max_pool_cuda(x)
        x = self.group_norm(x)
        return x