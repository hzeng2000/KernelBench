import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for group norm with scale
group_norm_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void group_norm_reduce_kernel(const float* x, float* sum_out, float* sum_sq_out, int batch, int groups, int C_per_group, int H, int W, int out_channels) {
    int b = blockIdx.x;
    int g = blockIdx.y;
    int c_start = g * C_per_group;
    int num_elements = C_per_group * H * W;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        int c_rel = i / (H * W);
        int h = (i % (H * W)) / W;
        int w = i % W;
        int c = c_start + c_rel;
        int idx = b * out_channels * H * W + c * H * W + h * W + w;
        float val = x[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }
    atomicAdd(&sum_out[b * groups + g], local_sum);
    atomicAdd(&sum_sq_out[b * groups + g], local_sum_sq);
}

__global__ void group_norm_normalize_kernel(const float* x, float* out, const float* mean, const float* inv_std, const float* weight, const float* bias, int batch, int groups, int C_per_group, int H, int W, int out_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * H * W) return;
    int b = idx / (out_channels * H * W);
    int c = (idx % (out_channels * H * W)) / (H * W);
    int h = (idx % (H * W)) / W;
    int w = idx % W;
    int g = c / C_per_group;
    float m = mean[b * groups + g];
    float inv_s = inv_std[b * groups + g];
    float val = x[idx];
    val = (val - m) * inv_s * weight[c] + bias[c];
    out[idx] = val;
}

torch::Tensor group_norm_scale_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups) {
    auto batch = x.size(0);
    auto out_channels = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    int C_per_group = out_channels / num_groups;
    auto sum = torch::zeros({batch * num_groups}, x.options());
    auto sum_sq = torch::zeros({batch * num_groups}, x.options());
    auto out = torch::zeros_like(x);
    
    const int block_size = 256;
    dim3 reduce_grid(batch, num_groups);
    group_norm_reduce_kernel<<<reduce_grid, block_size>>>(x.data_ptr<float>(), sum.data_ptr<float>(), sum_sq.data_ptr<float>(), batch, num_groups, C_per_group, H, W, out_channels);
    
    auto num_elements = C_per_group * H * W;
    auto mean = sum / num_elements;
    auto var = (sum_sq / num_elements) - mean * mean;
    const float eps = 1e-5f;
    auto inv_std = 1.0f / sqrtf(var + eps);
    
    int total_elements = batch * out_channels * H * W;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    group_norm_normalize_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), mean.data_ptr<float>(), inv_std.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), batch, num_groups, C_per_group, H, W, out_channels);
    
    return out;
}
"""

group_norm_scale_cpp_source = (
    "torch::Tensor group_norm_scale_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups);"
)

# Compile the inline CUDA code for group norm scale
group_norm_scale = load_inline(
    name="group_norm_scale",
    cpp_sources=group_norm_scale_cpp_source,
    cuda_sources=group_norm_scale_source,
    functions=["group_norm_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

# Define the custom CUDA kernel for maxpool with clamp
maxpool_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void maxpool_clamp_kernel(const float* x, float* out, int batch, int channels, int H, int W, int k, float clamp_min, float clamp_max) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int oh = blockIdx.z / (W / k);
    int ow = blockIdx.z % (W / k);
    int h_start = oh * k;
    int w_start = ow * k;
    float max_val = -INFINITY;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int h = h_start + i;
            int w = w_start + j;
            if (h < H && w < W) {
                int idx = b * channels * H * W + c * H * W + h * W + w;
                float val = x[idx];
                if (val > max_val) max_val = val;
            }
        }
    }
    max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
    int out_H = H / k;
    int out_W = W / k;
    int out_idx = b * channels * out_H * out_W + c * out_H * out_W + oh * out_W + ow;
    out[out_idx] = max_val;
}

torch::Tensor maxpool_clamp_cuda(torch::Tensor x, int k, float clamp_min, float clamp_max) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto out_H = H / k;
    auto out_W = W / k;
    auto out = torch::zeros({batch, channels, out_H, out_W}, x.options());
    
    dim3 grid(batch, channels, out_H * out_W);
    maxpool_clamp_kernel<<<grid, 1>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch, channels, H, W, k, clamp_min, clamp_max);
    
    return out;
}
"""

maxpool_clamp_cpp_source = (
    "torch::Tensor maxpool_clamp_cuda(torch::Tensor x, int k, float clamp_min, float clamp_max);"
)

# Compile the inline CUDA code for maxpool clamp
maxpool_clamp = load_inline(
    name="maxpool_clamp",
    cpp_sources=maxpool_clamp_cpp_source,
    cuda_sources=maxpool_clamp_source,
    functions=["maxpool_clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.group_norm_scale = group_norm_scale
        self.maxpool_clamp = maxpool_clamp

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        weight = self.scale.view(-1).to(x.device)
        bias = torch.zeros_like(weight)
        x = self.group_norm_scale.group_norm_scale_cuda(x, weight, bias, self.num_groups)
        x = self.maxpool_clamp.maxpool_clamp_cuda(x, self.maxpool_kernel_size, self.clamp_min, self.clamp_max)
        return x