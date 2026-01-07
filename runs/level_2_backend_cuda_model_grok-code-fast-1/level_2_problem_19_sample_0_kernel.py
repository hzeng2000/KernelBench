import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GELU and GroupNorm
fused_gelu_group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu(float x) {
    const float sqrt_2_pi = sqrtf(2.0f / M_PI);
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_pi * (x + 0.044715f * x3)));
}

__global__ void compute_sum_kernel(const float* x, float* sum, float* sum_sq, int N, int C, int H, int W, int num_groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int n = idx / (C * H * W);
    int c = (idx / (H * W)) % C;
    int g = c / (C / num_groups);
    float val = x[idx];
    float gelu_val = gelu(val);
    atomicAdd(&sum[n * num_groups + g], gelu_val);
    atomicAdd(&sum_sq[n * num_groups + g], gelu_val * gelu_val);
}

__global__ void normalize_kernel(const float* x, const float* mean, const float* var, const float* gamma, const float* beta, float* out, int N, int C, int H, int W, int num_groups, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int n = idx / (C * H * W);
    int c = (idx / (H * W)) % C;
    int g = c / (C / num_groups);
    float val = x[idx];
    float gelu_val = gelu(val);
    float m = mean[n * num_groups + g];
    float v = var[n * num_groups + g];
    float norm = (gelu_val - m) / sqrtf(v + eps);
    out[idx] = norm * gamma[c] + beta[c];
}

torch::Tensor fused_gelu_group_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int channels_per_group = C / num_groups;
    int num_elements = channels_per_group * H * W;
    
    auto sum_tensor = torch::zeros({N, num_groups}, torch::dtype(torch::kFloat32).device(x.device()));
    auto sum_sq_tensor = torch::zeros({N, num_groups}, torch::dtype(torch::kFloat32).device(x.device()));
    
    const int block_size = 256;
    int total_elements = N * C * H * W;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    compute_sum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), sum_tensor.data_ptr<float>(), sum_sq_tensor.data_ptr<float>(), N, C, H, W, num_groups);
    
    auto mean = sum_tensor / num_elements;
    auto var = sum_sq_tensor / num_elements - mean * mean;
    
    auto out = torch::zeros_like(x);
    
    normalize_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, num_groups, eps);
    
    return out;
}
"""

fused_gelu_group_norm_cpp_source = (
    "torch::Tensor fused_gelu_group_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps);"
)

# Compile the inline CUDA code for fused GELU and GroupNorm
fused_gelu_group_norm = load_inline(
    name="fused_gelu_group_norm",
    cpp_sources=fused_gelu_group_norm_cpp_source,
    cuda_sources=fused_gelu_group_norm_source,
    functions=["fused_gelu_group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution and fused GELU + GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.fused_gelu_group_norm = fused_gelu_group_norm
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_gelu_group_norm.fused_gelu_group_norm_cuda(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, 1e-5)
        return x