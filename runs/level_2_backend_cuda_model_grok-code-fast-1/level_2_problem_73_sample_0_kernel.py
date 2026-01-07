import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused batch norm and scaling
fused_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_sum_kernel(const float* x, float* sum, float* sum_sq, int channels, int num_per_channel, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int c = idx / num_per_channel;
    float val = x[idx];
    atomicAdd(&sum[c], val);
    atomicAdd(&sum_sq[c], val * val);
}

__global__ void normalize_kernel(const float* x, float* out, const float* mean, const float* var, const float* weight, const float* bias, float eps, int channels, int h, int w, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int n = idx / (channels * h * w);
    int c = (idx / (h * w)) % channels;
    float val = x[idx];
    float m = mean[c];
    float v = var[c];
    float w = weight[c];
    float b = bias[c];
    out[idx] = (val - m) / sqrtf(v + eps) * w + b;
}

torch::Tensor fused_bn_scale_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps, float scaling_factor, torch::Tensor running_mean, torch::Tensor running_var, float momentum, bool training) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto h = x.size(2);
    auto w = x.size(3);
    auto num_per_channel = batch * h * w;
    auto total_elements = x.numel();
    auto out = torch::zeros_like(x);
    auto sum = torch::zeros({channels}, x.options());
    auto sum_sq = torch::zeros({channels}, x.options());
    auto mean = torch::zeros({channels}, x.options());
    auto var = torch::zeros({channels}, x.options());

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    compute_sum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), sum.data_ptr<float>(), sum_sq.data_ptr<float>(), channels, num_per_channel, total_elements);

    mean = sum / num_per_channel;
    var = sum_sq / num_per_channel - mean * mean;

    if (training) {
        running_mean = running_mean * momentum + mean * (1 - momentum);
        running_var = running_var * momentum + var * (1 - momentum);
    }

    auto scaled_weight = weight * scaling_factor;
    auto scaled_bias = bias * scaling_factor;

    normalize_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), scaled_weight.data_ptr<float>(), scaled_bias.data_ptr<float>(), eps, channels, h, w, total_elements);

    return out;
}
"""

fused_bn_scale_cpp_source = (
    "torch::Tensor fused_bn_scale_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps, float scaling_factor, torch::Tensor running_mean, torch::Tensor running_var, float momentum, bool training);"
)

# Compile the inline CUDA code for fused batch norm and scaling
fused_bn_scale = load_inline(
    name="fused_bn_scale",
    cpp_sources=fused_bn_scale_cpp_source,
    cuda_sources=fused_bn_scale_source,
    functions=["fused_bn_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution and fused Batch Normalization with scaling using custom CUDA operators.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_bn_scale = fused_bn_scale

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_bn_scale.fused_bn_scale_cuda(x, self.bn.weight, self.bn.bias, self.bn.eps, self.scaling_factor, self.bn.running_mean, self.bn.running_var, self.bn.momentum, self.bn.training)
        return x