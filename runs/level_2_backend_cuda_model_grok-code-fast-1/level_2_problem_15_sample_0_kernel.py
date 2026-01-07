import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for BatchNorm and subtract mean
custom_bn_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void batch_norm_kernel(const float* x, const float* running_mean, const float* running_var, const float* weight, const float* bias, float* y, float* sums, int batch, int channels, int depth, int height, int width, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * depth * height * width;
    if (idx >= total) return;
    int b = idx / (channels * depth * height * width);
    int c = (idx / (depth * height * width)) % channels;
    float val = x[idx];
    float mean = running_mean[c];
    float var = running_var[c];
    float w = weight[c];
    float bi = bias[c];
    float norm = (val - mean) / sqrtf(var + eps);
    float out = norm * w + bi;
    y[idx] = out;
    atomicAdd(&sums[b * channels + c], out);
}

__global__ void subtract_mean_kernel(float* y, const float* means, int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * depth * height * width;
    if (idx >= total) return;
    int b = idx / (channels * depth * height * width);
    int c = (idx / (depth * height * width)) % channels;
    y[idx] -= means[b * channels + c];
}

torch::Tensor custom_bn_subtract_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float eps) {
    auto batch = x.size(0);
    auto channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);
    auto num_spatial = depth * height * width;
    auto y = torch::zeros_like(x);
    auto sums = torch::zeros({batch, channels}, x.options());
    const int block_size = 256;
    const int num_blocks = (x.numel() + block_size - 1) / block_size;
    batch_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), sums.data_ptr<float>(), batch, channels, depth, height, width, eps);
    auto means = sums / num_spatial;
    subtract_mean_kernel<<<num_blocks, block_size>>>(y.data_ptr<float>(), means.data_ptr<float>(), batch, channels, depth, height, width);
    return y;
}
"""

custom_bn_subtract_cpp_source = (
    "torch::Tensor custom_bn_subtract_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float eps);"
)

# Compile the inline CUDA code for custom BatchNorm and subtract mean
custom_bn_subtract = load_inline(
    name="custom_bn_subtract",
    cpp_sources=custom_bn_subtract_cpp_source,
    cuda_sources=custom_bn_subtract_source,
    functions=["custom_bn_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by custom CUDA BatchNorm and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.custom_bn_subtract = custom_bn_subtract

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.custom_bn_subtract.custom_bn_subtract_cuda(x, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight, self.batch_norm.bias, self.batch_norm.eps)
        return x