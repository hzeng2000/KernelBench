import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused group norm and global mean
fused_group_norm_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void group_mean_var_kernel(const float* input, float* sum_group, float* sumsq_group, int batch_size, int num_groups, int channels_per_group, int D, int H, int W) {
    extern __shared__ float shared_sum[];
    extern __shared__ float shared_sumsq[];
    
    int batch = blockIdx.z;
    int group = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int num_elements = channels_per_group * D * H * W;
    int idx = blockIdx.x * block_size + tid;
    
    int offset = batch * (num_groups * channels_per_group * D * H * W) + group * (channels_per_group * D * H * W);
    
    float val = 0.0f;
    if (idx < num_elements) {
        val = input[offset + idx];
    }
    shared_sum[tid] = val;
    shared_sumsq[tid] = val * val;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sumsq[tid] += shared_sumsq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int group_idx = batch * num_groups + group;
        atomicAdd(&sum_group[group_idx], shared_sum[0]);
        atomicAdd(&sumsq_group[group_idx], shared_sumsq[0]);
    }
}

__global__ void group_norm_and_sum_kernel(const float* input, float* sum_out, const float* sum_group, const float* sumsq_group, const float* gamma, const float* beta, int batch_size, int num_groups, int channels_per_group, int D, int H, int W, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_groups * channels_per_group * D * H * W;
    if (idx >= total_elements) return;
    
    int batch = idx / (num_groups * channels_per_group * D * H * W);
    int group = (idx / (channels_per_group * D * H * W)) % num_groups;
    int channel_in_group = (idx / (D * H * W)) % channels_per_group;
    int d = (idx / (H * W)) % D;
    int h = (idx / W) % H;
    int w = idx % W;
    
    int channel = group * channels_per_group + channel_in_group;
    int group_idx = batch * num_groups + group;
    
    float sum_g = sum_group[group_idx];
    float sumsq_g = sumsq_group[group_idx];
    int num_elements_group = channels_per_group * D * H * W;
    float mean = sum_g / num_elements_group;
    float var = sumsq_g / num_elements_group - mean * mean;
    
    float val = input[idx];
    float normalized = ((val - mean) / sqrtf(var + eps)) * gamma[channel] + beta[channel];
    
    atomicAdd(&sum_out[batch], normalized);
}

torch::Tensor fused_group_norm_mean_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, int batch_size, int out_channels, int D, int H, int W) {
    int channels_per_group = out_channels / num_groups;
    int num_elements_group = channels_per_group * D * H * W;
    int total_elements = batch_size * num_groups * num_elements_group;
    
    auto sum_group = torch::zeros({batch_size * num_groups}, torch::dtype(torch::kFloat32).device(x.device()));
    auto sumsq_group = torch::zeros({batch_size * num_groups}, torch::dtype(torch::kFloat32).device(x.device()));
    
    const int block_size = 256;
    dim3 grid_mean_var((num_elements_group + block_size - 1) / block_size, num_groups, batch_size);
    size_t shared_size = 2 * block_size * sizeof(float);
    group_mean_var_kernel<<<grid_mean_var, block_size, shared_size>>>(x.data_ptr<float>(), sum_group.data_ptr<float>(), sumsq_group.data_ptr<float>(), batch_size, num_groups, channels_per_group, D, H, W);
    
    auto sum_out = torch::zeros({batch_size}, torch::dtype(torch::kFloat32).device(x.device()));
    
    dim3 grid_norm_sum((total_elements + block_size - 1) / block_size, 1, 1);
    float eps = 1e-5f;
    group_norm_and_sum_kernel<<<grid_norm_sum, block_size>>>(x.data_ptr<float>(), sum_out.data_ptr<float>(), sum_group.data_ptr<float>(), sumsq_group.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), batch_size, num_groups, channels_per_group, D, H, W, eps);
    
    int total_num_elements = num_groups * num_elements_group;
    return sum_out / total_num_elements;
}
"""

fused_group_norm_mean_cpp_source = (
    "torch::Tensor fused_group_norm_mean_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, int batch_size, int out_channels, int D, int H, int W);"
)

# Compile the inline CUDA code for fused group norm and global mean
fused_group_norm_mean = load_inline(
    name="fused_group_norm_mean",
    cpp_sources=fused_group_norm_mean_cpp_source,
    cuda_sources=fused_group_norm_mean_source,
    functions=["fused_group_norm_mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies fused Group Normalization and global mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_op = fused_group_norm_mean

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x = self.conv(x)
        batch_size = x.shape[0]
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        return self.fused_op.fused_group_norm_mean_cuda(x, self.group_norm.weight, self.group_norm.bias, self.group_norm.num_groups, batch_size, self.group_norm.num_channels, D, H, W)