import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GroupNorm + LeakyReLU + scale by 2
fused_gn_leakyrelu_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_gn_leakyrelu_scale_kernel(const float* x, float* out, const float* gamma, const float* beta, int batch_size, int hidden_size, int num_groups, float eps, float negative_slope) {
    int group_size = hidden_size / num_groups;
    int batch = blockIdx.x;
    int group = blockIdx.y;
    int channel_in_group = threadIdx.x;
    if (channel_in_group >= group_size) return;
    int channel = group * group_size + channel_in_group;
    
    extern __shared__ float s_data[];
    float* s_sum = s_data;
    float* s_sum_sq = s_data + group_size;
    
    float val = x[batch * hidden_size + channel];
    s_sum[channel_in_group] = val;
    s_sum_sq[channel_in_group] = val * val;
    __syncthreads();
    
    for (int stride = 1; stride < group_size; stride *= 2) {
        if ((channel_in_group % (2 * stride) == 0) && (channel_in_group + stride < group_size)) {
            s_sum[channel_in_group] += s_sum[channel_in_group + stride];
            s_sum_sq[channel_in_group] += s_sum_sq[channel_in_group + stride];
        }
        __syncthreads();
    }
    
    float mean = s_sum[0] / group_size;
    float mean_sq = s_sum_sq[0] / group_size;
    float var = mean_sq - mean * mean;
    
    float normalized = (val - mean) / sqrtf(var + eps);
    float scaled = normalized * gamma[channel] + beta[channel];
    float activated = (scaled > 0.0f) ? scaled : (scaled * negative_slope);
    out[batch * hidden_size + channel] = activated * 2.0f;
}

torch::Tensor fused_gn_leakyrelu_scale_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps, float negative_slope) {
    auto batch_size = x.size(0);
    auto hidden_size = x.size(1);
    auto out = torch::zeros_like(x);
    
    int group_size = hidden_size / num_groups;
    dim3 blocks(batch_size, num_groups);
    dim3 threads(group_size);
    size_t shared_mem = 2 * group_size * sizeof(float);
    
    fused_gn_leakyrelu_scale_kernel<<<blocks, threads, shared_mem>>>(x.data_ptr<float>(), out.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), batch_size, hidden_size, num_groups, eps, negative_slope);
    
    return out;
}
"""

fused_gn_leakyrelu_scale_cpp_source = (
    "torch::Tensor fused_gn_leakyrelu_scale_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int num_groups, float eps, float negative_slope);"
)

# Compile the inline CUDA code for fused GroupNorm + LeakyReLU + scale
fused_gn_leakyrelu_scale = load_inline(
    name="fused_gn_leakyrelu_scale",
    cpp_sources=fused_gn_leakyrelu_scale_cpp_source,
    cuda_sources=fused_gn_leakyrelu_scale_source,
    functions=["fused_gn_leakyrelu_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, fused group normalization + leaky ReLU + scale by 2.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.fused_op = fused_gn_leakyrelu_scale
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.fc(x)
        x = self.fused_op.fused_gn_leakyrelu_scale_cuda(x, self.gn.weight, self.gn.bias, self.num_groups, self.eps, self.negative_slope)
        return x